"""
BART-Large-CNN Summarizer with Chunking
=======================================

Uses facebook/bart-large-cnn — one of the most popular and reliable 
summarization models on HuggingFace.

BART-Large-CNN has a max input of 1024 tokens. Since our retrieved context 
can be 2000–4000+ tokens, we use a chunking strategy:
  1. Split the input into overlapping chunks of ~900 tokens each
  2. Summarize each chunk independently
  3. If we get multiple chunk summaries, do a final "merge" pass 
     to combine them into one coherent answer
"""

import warnings
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "facebook/bart-large-cnn"


class BartChunkedSummarizer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.max_input_tokens = 1024  # BART's hard limit
        self.chunk_size = 900         # Leave some room for special tokens
        self.chunk_overlap = 100      # Overlap between chunks to preserve context
        self._load_model()

    def _load_model(self):
        print(f"\n{'='*60}")
        print(f"  Loading BART-Large-CNN Summarizer (with Chunking)")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}")
        print("  (First run downloads ~1.6GB — cached after that)")

        try:
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)

            print("  [2/2] Loading model weights...")
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()
            print("  ✅ BART Summarizer loaded successfully!")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"  ❌ Failed to load BART: {e}")
            print(f"{'='*60}\n")
            self.tokenizer = None
            self.model = None

    def _chunk_text(self, text: str) -> list:
        """
        Split text into overlapping chunks based on token count.
        Each chunk is at most `self.chunk_size` tokens, with `self.chunk_overlap` 
        tokens of overlap between consecutive chunks.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)

        if total_tokens <= self.chunk_size:
            # No chunking needed — text fits in one pass
            return [text]

        chunks = []
        start = 0
        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _summarize_single(self, text: str, max_length: int = 200, min_length: int = 30) -> str:
        """Summarize a single chunk that fits within BART's 1024-token limit."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=2,              # Light beam search (fast but better than greedy)
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize(self, context_text: str, query: str = "") -> str:
        """
        Summarize text using chunking strategy:
          1. Prepend the query to the context
          2. Split into overlapping chunks
          3. Summarize each chunk
          4. If multiple chunks → merge summaries in a final pass
        """
        if not self.model or not self.tokenizer:
            return "❌ BART model is not loaded."

        # Prepend query for query-focused summarization
        if query:
            input_text = f"Question: {query}\n\nContext:\n{context_text}"
        else:
            input_text = context_text

        # Step 1: Chunk the text
        chunks = self._chunk_text(input_text)
        num_chunks = len(chunks)

        print(f"\n  [Summarizer] Input split into {num_chunks} chunk(s)")

        # Step 2: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  [Chunk {i}/{num_chunks}] Summarizing ({len(self.tokenizer.encode(chunk))} tokens)...")
            summary = self._summarize_single(chunk, max_length=200, min_length=30)
            chunk_summaries.append(summary)

        # Step 3: If only one chunk, return directly
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        # Step 4: Merge multiple chunk summaries into one final summary
        merged_text = " ".join(chunk_summaries)
        merged_tokens = len(self.tokenizer.encode(merged_text))

        if merged_tokens <= self.chunk_size:
            # Merged text fits in one pass — do a final refinement
            print(f"  [Final Pass] Merging {len(chunk_summaries)} chunk summaries ({merged_tokens} tokens)...")
            final_summary = self._summarize_single(merged_text, max_length=300, min_length=50)
            return final_summary
        else:
            # Very rare: even merged summaries are too long, just concatenate
            print(f"  [Merge] Concatenating {len(chunk_summaries)} summaries (too long for final pass)")
            return " ".join(chunk_summaries)
