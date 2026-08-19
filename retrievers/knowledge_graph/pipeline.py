"""
Complete RAG Pipeline (Retrieval-Augmented Generation)
======================================================

The final entry point for the Mental Health Chatbot knowledge system.

Flow:
    1. Query → Fusion Retriever
    2. Fusion Retriever pulls Top 5 from Books + Top 5 from Reddit QA
    3. 10 Contexts → Abstractive Summarizer
    4. Summarizer outputs final synthesized response

Usage:
    python -m knowledge.pipeline
"""

import sys
from knowledge.retrieve import HybridRetriever

def main():
    print("\nInitializing Mental Health Book Retrieval Pipeline...")
    print("=" * 60)
    
    try:
        retriever = HybridRetriever()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have run the ingest scripts:")
        print("  python -m knowledge.ingest")
        print("  python -m knowledge.ingest_embeddings")
        return

    print("=" * 60)
    print("Retriever Ready! (Type 'quit' or 'exit' to stop)")
    print("=" * 60)

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query.strip():
                continue

            # 1. Retrieve Contexts (Books Only)
            contexts = retriever.retrieve_top_k(query, k=5, similarity_pool=10)
            
            if not contexts:
                print("Bot: I couldn't find any relevant information.")
                continue
                
            # 2. Display Output
            print("\nTop 5 Retrieved Paragraphs (Ranked by MMR):")
            print("-" * 50)
            
            for i, ctx in enumerate(contexts, 1):
                print(f"[{i}] Source: {ctx['source']}")
                print(f"    Relevance Score: {ctx['relevance']:.4f} | Final MMR: {ctx['score']:.4f}")
                print(f"    {ctx['text'].strip()}")
                print("-" * 60)
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()


