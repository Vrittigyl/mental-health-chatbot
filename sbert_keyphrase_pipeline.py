"""
SBERT Keyphrase Similarity Retrieval & Summarization Pipeline
============================================================

1. Converts input query into SBERT embedding (`all-MiniLM-L6-v2`).
2. Compares with `train_keyphrase_embeddings.pt` (or `test_keyphrase_embeddings.pt`).
3. Retrieves Top 5 matching comments/answers from dataset (`train_dataset.jsonl`).
4. Extracts dialog analysis features: Emotion, Severity, Intent, Cause.
5. Merges features & retrieved comments and generates abstractive response via AbstractiveSummarizer (Ollama Gemma3).

Usage:
    python sbert_keyphrase_pipeline.py
"""

import os
import sys
import time
import json
import warnings
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Summarizer
try:
    from summarizer import AbstractiveSummarizer
except ImportError:
    AbstractiveSummarizer = None

# Global Model Initializations
print("Initializing SBERT Model (all-MiniLM-L6-v2)...")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load Dialog Analysis components
print("Loading Emotion Service...")
try:
    from metrics.emotion.service import EmotionService
    emotion_service = EmotionService()
except Exception as e:
    print(f"Warning: EmotionService not loaded: {e}")
    emotion_service = None

print("Loading Severity Detector...")
try:
    from metrics.severity import detect_severity
except Exception as e:
    print(f"Warning: Severity detector not loaded: {e}")
    detect_severity = None

print("Loading Intent Extractor...")
try:
    from metrics.intent import detect_intent
except Exception as e:
    print(f"Warning: Intent extractor not loaded: {e}")
    extract_intent = None

print("Loading Cause Extractor...")

cause_bosch_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cause_bosch"
)

if cause_bosch_path not in sys.path:
    sys.path.append(cause_bosch_path)

try:
    from metrics.cause_bosch.extract import CauseEffectExtractor

    cause_extractor = CauseEffectExtractor(
        model_dir=".",
        base_model_name="roberta-large",
    )
except Exception as e:
    print(f"Warning: Cause extractor (Bosch) not loaded: {e}")
    cause_extractor = None

class SBERTRetrieverPipeline:
    def __init__(self, dataset_path="train_dataset.jsonl", embeddings_path="train_keyphrase_embeddings.pt"):
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        
        print(f"\n--- Loading Embeddings & Dataset ({dataset_path}) ---")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        t0 = time.time()
        print(f"Loading embedding tensor from {embeddings_path}...")
        self.embeddings = torch.load(embeddings_path, map_location="cpu")
        print(f"Loaded embeddings tensor shape: {self.embeddings.shape} in {time.time()-t0:.2f}s")
        
        t0 = time.time()
        print(f"Loading dataset records from {dataset_path}...")
        self.df = pd.read_json(dataset_path, lines=True)
        print(f"Loaded {len(self.df)} records in {time.time()-t0:.2f}s")
        
        print("Loading Abstractive Summarizer...")
        try:
            self.summarizer = AbstractiveSummarizer() if AbstractiveSummarizer else None
        except Exception as e:
            print(f"Warning: Could not load AbstractiveSummarizer: {e}")
            self.summarizer = None

    def analyze_dialog(self, query: str) -> dict:
        """Extracts emotion, severity, intent, and cause from query."""
        results = {
            "emotion": "None",
            "severity": "None",
            "severity_score": 0.0,
            "intent": {"action": "", "object": ""},
            "cause": [],
            "effect": [],
            "signal": []
        }
        
        # 1. Emotion
        if emotion_service:
            try:
                results["emotion"] = emotion_service.get_primary_emotion(query)
            except Exception as e:
                print(f"Emotion extraction error: {e}")

        # 2. Severity
        if detect_severity:
            try:
                sev_label, sev_score = detect_severity(query)
                results["severity"] = sev_label
                results["severity_score"] = float(sev_score)
            except Exception as e:
                print(f"Severity extraction error: {e}")

        # 3. Intent
        if extract_intent:
            try:
                act, obj = detect_intent(query)
                results["intent"] = {"action": act or "", "object": obj or ""}
            except Exception as e:
                print(f"Intent extraction error: {e}")

        # 4. Cause
        if cause_extractor:
            try:
                cause_results = cause_extractor.predict([query])[0]

                causes = []
                effects = []
                signals = []

                for rel in cause_results:
                    if rel.get("cause"):
                        causes.append(str(rel["cause"]))
                    if rel.get("effect"):
                        effects.append(str(rel["effect"]))
                    if rel.get("signal"):
                        signals.append(str(rel["signal"]))

                results["cause"] = causes
                results["effect"] = effects
                results["signal"] = signals

            except Exception as e:
                print(f"Cause extraction error: {e}")

        return results

    def retrieve_top_comments(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Encodes query with SBERT, computes cosine similarity against stored keyphrase embeddings,
        and retrieves top matching comments/answers.
        """
        print(f"\nEncoding query with SBERT...")
        query_emb = sbert_model.encode(query, convert_to_tensor=True, normalize_embeddings=True).cpu()
        
        # Cosine similarity matrix multiplication (since embeddings are normalized)
        print("Computing similarity against keyphrase embeddings...")
        similarities = torch.matmul(self.embeddings, query_emb)
        
        top_scores, top_indices = torch.topk(similarities, k=min(top_k * 5, len(self.df)))
        
        retrieved = []
        seen_comments = set()
        
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            row = self.df.iloc[idx]
            disease = row.get("disease", "Unknown")
            question = str(row.get("question(post)", "")).strip()
            comments = row.get("answers(comments)", [])
            
            if isinstance(comments, str):
                comments = [comments]
                
            for comment in comments:
                comment_clean = str(comment).strip()
                if not comment_clean or comment_clean in seen_comments:
                    continue
                seen_comments.add(comment_clean)
                
                retrieved.append({
                    "dataset_idx": idx,
                    "similarity": round(float(score), 4),
                    "disease": disease,
                    "question_snippet": question[:150] + "..." if len(question) > 150 else question,
                    "text": comment_clean,
                    "source": f"Dataset Post #{idx} (Disease: {disease}, Similarity: {score:.3f})"
                })
                
                if len(retrieved) >= top_k:
                    break
            if len(retrieved) >= top_k:
                break
                
        return retrieved

    def process_query(self, query: str, top_k: int = 5) -> dict:
        """Runs end-to-end pipeline for a given query."""
        print(f"\n{'='*60}")
        print(f"🔍 Processing Query: '{query}'")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Extract Dialog Features
        print("\n--- Step 1: Running Dialog Analysis ---")
        dialog_analysis = self.analyze_dialog(query)
        print(f"  Emotion  : {dialog_analysis['emotion']}")
        print(f"  Severity : {dialog_analysis['severity']} (Score: {dialog_analysis['severity_score']:.4f})")
        print(f"  Intent   : Action='{dialog_analysis['intent']['action']}', Object='{dialog_analysis['intent']['object']}'")
        print(f"  Cause    : {', '.join(dialog_analysis['cause']) if dialog_analysis['cause'] else 'None'}")
        
        # Step 2: Retrieve Top Comments/Answers using SBERT Embeddings
        print(f"\n--- Step 2: SBERT Similarity Search & Retrieval (Top {top_k}) ---")
        retrieved_comments = self.retrieve_top_comments(query, top_k=top_k)
        
        print("\nRetrieved Comments/Answers:")
        for i, item in enumerate(retrieved_comments, 1):
            print(f" [{i}] {item['source']}")
            print(f"     Comment: {item['text'][:200]}...")
            
        # Step 3: Combine Context
        combined_context = {
            "query": query,
            "dialog_analysis": dialog_analysis,
            "knowledge_contexts": retrieved_comments
        }
        
        # Step 4: Summarize Response using Abstractive Summarizer
        summary_response = ""
        if self.summarizer:
            print("\n--- Step 3: Generating Response via Abstractive Summarizer ---")
            summary_response = self.summarizer.generate_response(combined_context)
        else:
            summary_response = "[Summarizer model not available. Returning retrieved comments and dialog analysis.]"

        elapsed = time.time() - start_time
        print(f"\n⏱️ Total Processing Time: {elapsed:.2f}s")
        
        return {
            "query": query,
            "dialog_analysis": dialog_analysis,
            "retrieved_comments": retrieved_comments,
            "combined_context": combined_context,
            "summary_response": summary_response
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SBERT Keyphrase Retrieval and Summarization Pipeline")
    parser.add_argument("--use_test", action="store_true", help="Use test dataset/embeddings instead of train dataset")
    args = parser.parse_args()

    if args.use_test:
        dataset_path = "test_dataset.jsonl"
        embeddings_path = "test_keyphrase_embeddings.pt"
    else:
        dataset_path = "train_dataset.jsonl"
        embeddings_path = "train_keyphrase_embeddings.pt"

    pipeline = SBERTRetrieverPipeline(dataset_path=dataset_path, embeddings_path=embeddings_path)

    print("\n🤖 Pipeline Ready! (Type 'quit' or 'exit' to stop)")
    print("=" * 60)

    while True:
        try:
            query = input("\n👤 Enter Query: ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            if not query:
                continue

            result = pipeline.process_query(query, top_k=5)

            print("\n" + "═" * 60)
            print("💡 FINAL RESPONSE")
            print("═" * 60)
            print(result["summary_response"])
            print("═" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
