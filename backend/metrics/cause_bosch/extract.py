"""
Standalone inference for the BoschAI CNC Subtask-2 cause/effect/signal
extraction model (https://github.com/boschresearch/boschai-cnc-shared-task-ranlp2023).

Usage:
    Make sure `models/` and `utils/` from the CNC repo are importable
    (either copy those two folders next to this script, or add the
    cloned repo root to PYTHONPATH).

    from cnc_st2_inference import CauseEffectExtractor

    extractor = CauseEffectExtractor(
        model_dir="path/to/output/st2/best_model",  # folder containing model.pt
        base_model_name="roberta-large",             # whatever you passed as --model_name_or_path
    )
    results = extractor.predict(["I feel exhausted because I haven't slept in days."])
    print(results)
"""

from __future__ import annotations

import os
from typing import List, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoTokenizer

# These come from the CNC repo — copy `models/` and `utils/` into your project,
# or add the cloned repo to PYTHONPATH.
from metrics.cause_bosch.models.bilou_tagger_st2 import NER_CRF_Classifier
from metrics.cause_bosch.utils.bilou_tags import THREE_LAYER_BILOU_TAGS, tl_bilou_id2ne_label

MAX_LENGTH = 512


class Relation(TypedDict):
    cause: Optional[str]
    effect: Optional[str]
    signal: Optional[str]


def _build_crf_mask(tokenized_inputs, tokens: List[str], batch_index: int) -> List[int]:
    """Marks the first subword token of every whitespace-split word (matches training code)."""
    word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
    word2tok = {w: [] for w in range(len(tokens))}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None:
            word2tok[word_idx].append(token_idx)

    crf_mask = [0] * MAX_LENGTH
    for j in range(1, len(word2tok) + 1):
        if len(word2tok[j - 1]) == 0:
            continue
        crf_mask[word2tok[j - 1][0]] = 1
    return crf_mask


def _extract_relations_for_layers(tokens: List[str], layer_tags: List[List[str]]) -> List[Relation]:
    """
    Re-implementation of the span-finding logic in train_st2.py's get_final_sentences,
    but returning the actual substrings instead of an annotated sentence.
    Each of the (up to) 3 layers can hold at most one Cause/Effect/Signal relation.
    """
    relations: List[Relation] = []

    for pl in layer_tags:
        if all(tag == "O" for tag in pl):
            continue

        def find_span(arg_name: str) -> Optional[str]:
            begin = end = -1
            found = False
            for i in range(len(pl)):
                if f"U-{arg_name}" in pl[i]:
                    begin = end = i
                    found = True
                    break
                elif f"B-{arg_name}" in pl[i]:
                    begin = i
                    j = i + 1
                    while j < len(pl):
                        if f"I-{arg_name}" in pl[j]:
                            j += 1
                            continue
                        elif f"L-{arg_name}" in pl[j]:
                            end = j
                            found = True
                            break
                        else:
                            break
                    break
            if not found or begin == -1 or end == -1:
                return None
            return " ".join(tokens[begin : end + 1])

        relations.append(
            {
                "cause": find_span("ARG0"),
                "effect": find_span("ARG1"),
                "signal": find_span("SIG0"),
            }
        )

    return relations


class CauseEffectExtractor:
    def __init__(self, model_dir: str, base_model_name: str = "roberta-large", device: Optional[str] = None):
        """
        Args:
            model_dir: directory containing `model.pt` (the state_dict saved by your patched script).
            base_model_name: the --model_name_or_path used during training (e.g. "roberta-large").
            device: "cuda" or "cpu"; auto-detected if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(base_model_name)
        if config.model_type in {"gpt2", "roberta"}:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, use_fast=True, add_prefix_space=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

        self.model = NER_CRF_Classifier(base_model_name, THREE_LAYER_BILOU_TAGS)
        state_dict_path = os.path.join(model_dir, "model.pt")
        state_dict = torch.load(state_dict_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentences: List[str]) -> List[List[Relation]]:
        """
        Args:
            sentences: raw, un-tokenized input sentences.

        Returns:
            One list of relations per input sentence. Each relation is a dict with
            "cause", "effect", "signal" keys (values are strings or None).
        """
        all_tokens = [s.split() for s in sentences]

        tokenized_inputs = self.tokenizer(
            all_tokens,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        if "token_type_ids" not in tokenized_inputs.data:
            tokenized_inputs["token_type_ids"] = [
                [0] * MAX_LENGTH for _ in range(len(all_tokens))
            ]

        crf_masks = [
            _build_crf_mask(tokenized_inputs, tokens, i) for i, tokens in enumerate(all_tokens)
        ]

        input_ids = torch.tensor(tokenized_inputs["input_ids"]).to(self.device)
        attention_mask = torch.tensor(tokenized_inputs["attention_mask"]).to(self.device)
        token_type_ids = torch.tensor(tokenized_inputs["token_type_ids"]).to(self.device)
        crf_mask = torch.tensor(crf_masks).to(self.device)
        sorted_crf_mask = torch.sort(crf_mask, dim=1, descending=True, stable=True)[0]

        with torch.no_grad():
            predictions = self.model.predict_tag_sequence(
                input_ids, attention_mask, token_type_ids, crf_mask, sorted_crf_mask
            )

        results: List[List[Relation]] = []
        for tokens, pred in zip(all_tokens, predictions):
            layer_tags: List[List[str]] = [[], [], []]
            for p in pred:
                l1, l2, l3 = tl_bilou_id2ne_label[p].split("|")
                layer_tags[0].append(l1)
                layer_tags[1].append(l2)
                layer_tags[2].append(l3)
            # predictions are per-word (crf_mask picks first subword of each word),
            # so they should line up 1:1 with `tokens`
            layer_tags = [layer[: len(tokens)] for layer in layer_tags]
            results.append(_extract_relations_for_layers(tokens, layer_tags))

        return results


if __name__ == "__main__":
    extractor = CauseEffectExtractor(
        model_dir=".",
        base_model_name="roberta-large",
    )
    examples = [
        """ I'm an adult dyslexic in college, about to finish my Bachelor's degree.  Up until this point, I have been able to compensate for my disability to the point that no one had a clue about my dyslexia.  I would study morning, noon, and night for every class just to get through the reading and to prepare for tests.  
     But there were definitely some indicators early on:  Even though I learned to read when I was 4, I still was in the slowest reading level and my problems with numbers were so bad that I didn't understand how to tell time on a non-digital clock until I was about 12.  The only subject I ever had major problems with was Math--It seemed like no matter how much I studied, how much tutoring I got, I could never understand the concepts.  To this day I still cannot perform basic math functions in my head.  But until now, I've always thought dyslexia had to do with reading backwards and that's it.  I thought everything else I was experiencing was me just having below average intelligence....until I talked to a speech pathologist who informed me about some other symptoms.   
     Now, I am an adult and it is getting to a point that I cannot compensate any longer.  The textbooks are nearly impossible for me to read, not only because of the language but also because of the font, the stark white pages, and all the numbers and formulas that are included.  In classes, the professors lecture in a way that I cannot understand what they are saying because they are so unorganized and go off on irrelevant tangents that are hard to follow.  And when they ask me questions that I have problems understanding, or put me on the spot where I have to do math in my head, I just babble and start panicking because I'm so confused.  My professors and other students are getting impatient with me and several times I have gotten made fun of or yelled at in front of the class.  And it is always about things that relate to Dyslexia.  
     I have tried colored overlays, different reading tactics, etc. and nothing seems to make a noticeable difference.  I looked into getting diagnosed but it is very expensive if you are an adult.  My insurance doesn't cover any of the psychologists that perform the testing in my area.  Not that it really matters, all it would get me is a longer time for test taking, which is useful but doesn't help me actually learn the stuff.  I'm getting to a point that I want to just give up.  I'm hoping someone out there can relate to my frustrations or have advice on how to tell my professors in a way that they won't think I'm just trying to get special treatment."""
    ]
    for sentence, relations in zip(examples, extractor.predict(examples)):
        print(sentence)
        for r in relations:
            print("  ", r)