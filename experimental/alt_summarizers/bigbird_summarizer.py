"""
BigBird-Pegasus Abstractive Summarizer (PubMed)
================================================

Uses google/bigbird-pegasus-large-pubmed for abstractive summarization
of retrieved medical textbook paragraphs.

Why BigBird-Pegasus-PubMed?
  - Supports up to 4096 tokens input (vs 512 for regular Pegasus/BERT)
  - Fine-tuned on PubMed biomedical literature → perfect for mental health content
  - Generates fluent, medically accurate abstractive summaries
  - Runs 100% locally on CPU — no API keys, no internet needed after download

Architecture:
  - Encoder: BigBird sparse attention (handles long documents efficiently)
  - Decoder: Pegasus-style autoregressive generation
  - Tokenizer: SentencePiece (subword tokenization)
"""

import warnings
import torch
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Model identifier on Hugging Face
MODEL_NAME = "google/bigbird-pegasus-large-pubmed"


class BigBirdSummarizer:
    """
    Abstractive summarizer using BigBird-Pegasus fine-tuned on PubMed.

    This model can accept up to 4096 tokens as input, which means we can
    feed all 5 retrieved paragraphs (~1500 words) without any truncation.

    The model generates a concise, medically-grounded summary that synthesizes
    information from all the retrieved paragraphs into a coherent response.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load_model()

    def _load_model(self):
        """Download (first time only) and load the model + tokenizer."""
        print(f"\n{'='*60}")
        print(f"  Loading BigBird-Pegasus-PubMed Summarizer")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}")
        print("  (First run downloads ~3.5GB — cached after that)")

        try:
            print("  [1/2] Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            print("  [2/2] Loading model weights...")
            self.model = BigBirdPegasusForConditionalGeneration.from_pretrained(
                self.model_name,
                attention_type="block_sparse",  # Sparse attention is required for fast 4000+ token processing!
            )
            self.model.eval()  # Set to evaluation mode (no dropout)
            print("  ✅ Summarizer loaded successfully!")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"  ❌ Failed to load summarizer: {e}")
            print(f"{'='*60}\n")
            self.tokenizer = None
            self.model = None

    def summarize(self, context_text: str, query: str = "") -> str:
        """
        Generate an abstractive summary from the retrieved paragraphs.

        Args:
            context_text: Combined text from retrieved paragraphs.
            query:        The original user query (prepended for query-focused summarization).

        Returns:
            A concise, medically-grounded summary string.
        """
        if not self.model or not self.tokenizer:
            return "❌ Summarizer model is not loaded. Please check the error above."

        # Prepend the query for query-focused summarization
        # This guides the model to focus on what the user actually asked
        if query:
            input_text = f"Question: {query}\n\nContext: {context_text}"
        else:
            input_text = context_text

        print("\n  [Summarizer] Generating abstractive response...")

        try:
            # Tokenize with BigBird's 4096 token limit to fit all retrieved paragraphs
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=4096,      # Restored to 4096 to prevent truncation
                truncation=True,
                padding=True,
            )

            # Generate summary (Optimized for CPU speed)
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,       # Max output tokens (reduced for speed)
                    min_length=30,        # Minimum output length
                    num_beams=1,          # Changed to 1 (Greedy search) - MUCH faster than beam search=4
                    length_penalty=1.0,   
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            # Decode the generated tokens back to text
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary

        except Exception as e:
            return f"❌ Error during summarization: {e}"
