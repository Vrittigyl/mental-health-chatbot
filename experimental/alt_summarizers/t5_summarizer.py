"""
Long-T5 Summarizer (PubMed / BookSum)
=====================================

Uses pszemraj/long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP
A powerful LongT5 model that can handle up to 16,384 tokens of input context,
making it perfect for combining textbook paragraphs and reddit Q&A discussions
into a single prompt.
"""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# We are switching to the stable "base" version. The "large-WIP" version you requested 
# is currently broken on HuggingFace (missing embedding weights, generating gibberish) 
# and is too heavy for CPU inference, causing the terminal to hang.
MODEL_NAME = "pszemraj/long-t5-tglobal-base-16384-book-summary"


class LongT5Summarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"\n{'='*60}")
        print(f"  Loading Long-T5 Summarizer")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}")
        print("  (First run downloads ~3.1GB — cached after that)")

        try:
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print("  [2/2] Loading model weights...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            print("  ✅ Summarizer loaded successfully!")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"  ❌ Failed to load summarizer: {e}")
            print(f"{'='*60}\n")
            self.tokenizer = None
            self.model = None

    def summarize(self, context_text: str, query: str = "") -> str:
        if not self.model or not self.tokenizer:
            return "❌ Summarizer model is not loaded."

        # The model is trained to summarize text. We can format it nicely.
        if query:
            input_text = f"Summarize the following context to answer the question.\n\nQuestion: {query}\n\nContext:\n{context_text}"
        else:
            input_text = f"Summarize the following text:\n\n{context_text}"

        print("\n  [Summarizer] Generating abstractive response...")

        try:
            # Tokenize with 16384 limit (though we cap at 4096 for CPU speed sanity)
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=4096,
                truncation=True,
                padding=True,
            )

            # Generate summary (Optimized for CPU speed)
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=200,       # Max output tokens
                    min_length=30,        # Minimum output length
                    num_beams=1,          # Greedy search for fast CPU inference
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            return f"❌ Error during summarization: {e}"
