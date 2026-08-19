"""
Pegasus-X Summarizer
====================

Uses google/pegasus-x-base (or large). Pegasus-X is optimized for extremely long contexts (up to 16,384 tokens) 
using Staggered Block attention, making it perfect for combining textbook and Reddit data.
"""

import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Fine-tuned on BookSum by pszemraj — should produce coherent summaries
MODEL_NAME = "pszemraj/pegasus-x-large-book-summary"


class PegasusXSummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"\n{'='*60}")
        print(f"  Loading Pegasus-X Summarizer")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}")

        try:
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print("  [2/2] Loading model weights...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.eval()  # Set to evaluation mode
            print("  ✅ Pegasus-X Summarizer loaded successfully!")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"  ❌ Failed to load Pegasus-X: {e}")
            print(f"{'='*60}\n")
            self.tokenizer = None
            self.model = None

    def summarize(self, context_text: str, query: str = "") -> str:
        if not self.model or not self.tokenizer:
            return "❌ Pegasus-X model is not loaded."

        # Seq2Seq formatting
        if query:
            input_text = f"Question: {query}\n\nContext:\n{context_text}"
        else:
            input_text = f"Context:\n{context_text}"

        print("\n  [Summarizer] Generating abstractive response with Pegasus-X...")

        try:
            # Tokenize with 16384 limit (Pegasus-X's specialty)
            # We cap it at 4096 here for CPU speed, but you can increase it to 16384
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
                    max_length=512,       # Increased so the model can finish its sentences
                    min_length=50,        # Minimum output length
                    num_beams=1,          # Greedy search for fast CPU inference
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            return f"❌ Error during summarization: {e}"
