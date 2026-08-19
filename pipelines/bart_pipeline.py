"""
End-to-End RAG Pipeline with BART-Large-CNN (Chunked) Summarizer
================================================================

Full pipeline:
  1. User types a question
  2. HybridRetriever searches 5 medical textbooks
  3. Bert/retrieve searches the Reddit Q&A dataset
  4. Both contexts are combined
  5. BART-Large-CNN summarizes using overlapping chunks for long inputs
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from retrievers.book_retriever import HybridRetriever, clean_output_text
from retrievers.reddit_retriever import load_training_data, find_similar
from summarizers.bart_summarizer import BartChunkedSummarizer


def main():
    print("\n" + "=" * 70)
    print("  🧠 Mental Health Chatbot — RAG (Books + Reddit) + BART-Large-CNN")
    print("=" * 70)

    # ── Step 1: Load Knowledge Retriever (Textbooks) ──
    print("\n📚 Loading Knowledge Retriever (Textbooks)...")
    try:
        book_retriever = HybridRetriever()
        print("✅ Textbook Retriever loaded!\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # ── Step 2: Load Reddit Q&A Data ──
    print("💬 Loading Reddit Q&A Dataset...")
    try:
        reddit_data = load_training_data()
        if reddit_data:
            print("✅ Reddit Dataset loaded!\n")
        else:
            print("⚠️ Reddit Dataset is empty. Continuing without it.\n")
    except Exception as e:
        print(f"❌ Error loading Reddit data: {e}")
        reddit_data = []

    # ── Step 3: Load Summarizer (BART with Chunking) ──
    try:
        summarizer = BartChunkedSummarizer()
    except Exception as e:
        print(f"❌ Could not load summarizer: {e}")
        summarizer = None

    # ── Step 4: Interactive Chat Loop ──
    print("=" * 70)
    print("  Chatbot is ready! Type your question below.")
    print("  Type 'quit' to exit.")
    print("=" * 70)

    while True:
        try:
            print("\n" + "-" * 70)
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            print("\nGoodbye! 👋")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        combined_contexts = []

        # ── A. Retrieve from Textbooks ──
        print("\n  🔍 Searching Textbooks...")
        book_results = book_retriever.retrieve_top_k(user_input, k=5, similarity_pool=10)
        
        if book_results:
            combined_contexts.append("=== MEDICAL TEXTBOOK EXCERPTS ===")
            print(f"  📖 Found {len(book_results)} textbook paragraphs.")
            for i, result in enumerate(book_results, 1):
                cleaned = clean_output_text(result["text"])
                combined_contexts.append(f"[Textbook: {result['source']}]\n{cleaned}\n")
        else:
            print("  ⚠️ No relevant textbook paragraphs found.")

        # ── B. Retrieve from Reddit Q&A ──
        if reddit_data:
            print("  🔍 Searching Reddit Discussions...")
            query_emb = book_retriever.model.encode(user_input)
            reddit_results = find_similar(query_emb, reddit_data, top_k=5)
            
            if reddit_results:
                combined_contexts.append("=== REDDIT DISCUSSIONS ===")
                print(f"  💬 Found {len(reddit_results)} Reddit discussions.")
                for i, result in enumerate(reddit_results, 1):
                    q = result["question"]
                    ans = result["answers"][0]["answer"] if result["answers"] else "No answer available."
                    combined_contexts.append(f"[Reddit Discussion - {result['disease']}]\nQuestion: {q}\nTop Answer: {ans}\n")
            else:
                print("  ⚠️ No relevant Reddit discussions found.")

        # ── C. Combine everything ──
        if not combined_contexts:
            print("\n  🤖 Answer: I couldn't find any relevant information in my databases to answer that.")
            continue

        full_context_string = "\n".join(combined_contexts)

        # ── D. Show Combined Context in Terminal ──
        print("\n" + "=" * 70)
        print("  RAW EXTRACTED CONTEXT (Sent to Summarizer):")
        print("=" * 70)
        print(f"\n{full_context_string}\n")
        print("=" * 70)

        # ── E. Generate abstractive summary ──
        if summarizer:
            summary = summarizer.summarize(full_context_string, query=user_input)

            print("\n" + "=" * 70)
            print("  🤖 Answer:")
            print("=" * 70)
            print(f"\n  {summary}")
            print("\n" + "=" * 70)
        else:
            print("\n  ⚠️ Summarizer not loaded. Showing raw data only:")
            print(full_context_string)


if __name__ == "__main__":
    main()
