import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from retrievers.book_retriever import HybridRetriever, clean_output_text
from retrievers.reddit_retriever import load_training_data, find_similar
from summarizers.gemma_summarizer import GemmaSummarizer
from pipeline_test import analyze_query

def main():
    print("\n" + "=" * 70)
    print("  🌟 Mental Health Chatbot — Final Integrated Pipeline")
    print("  (Emotion/Cause/Severity + Textbooks + Reddit + Gemma LLM)")
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

    # ── Step 3: Load Summarizer (Gemma via Ollama) ──
    try:
        summarizer = GemmaSummarizer()
    except Exception as e:
        print(f"❌ Could not load Gemma summarizer: {e}")
        summarizer = None

    # ── Step 4: Interactive Chat Loop ──
    print("=" * 70)
    print("  Chatbot is ready! Type your question below.")
    print("  Press Enter on an empty line, or type 'quit' to exit.")
    print("=" * 70)

    while True:
        try:
            print("\n" + "-" * 70)
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        # Exit on empty input or 'quit'
        if not user_input:
            print("\nGoodbye! 👋")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        # ── A. Analyze the User's Query ──
        # This calls your pipeline.py function to extract emotion, intent, etc.
        analysis = analyze_query(user_input)
        
        # ── B. Retrieve from Textbooks ──
        combined_contexts = []
        print("\n  🔍 Searching Textbooks...")
        book_results = book_retriever.retrieve_top_k(user_input, k=3, similarity_pool=10)
        
        if book_results:
            combined_contexts.append("=== MEDICAL TEXTBOOK EXCERPTS ===")
            for result in book_results:
                cleaned = clean_output_text(result["text"])
                combined_contexts.append(f"[Textbook: {result['source']}]\n{cleaned}\n")

        # ── C. Retrieve from Reddit Q&A ──
        if reddit_data:
            print("  🔍 Searching Reddit Discussions...")
            query_emb = book_retriever.model.encode(user_input)
            reddit_results = find_similar(query_emb, reddit_data, top_k=3)
            
            if reddit_results:
                combined_contexts.append("=== REDDIT DISCUSSIONS ===")
                for result in reddit_results:
                    q = result["question"]
                    ans = result["answers"][0]["answer"] if result["answers"] else "No answer available."
                    if len(ans) > 500:
                        ans = ans[:500] + "..."
                    combined_contexts.append(f"[Reddit Discussion - {result['disease']}]\nQuestion: {q}\nTop Answer: {ans}\n")

        # ── D. Prepare Prompt for Gemma ──
        if not combined_contexts:
            print("\n  🤖 Answer: I couldn't find any relevant information to answer that.")
            continue

        full_context_string = "\n".join(combined_contexts)

        # Build a powerful query string that forces the LLM to consider the user's emotional state
        enhanced_query = f"{user_input}\n\n[User's Current Mental State Profile]"
        if analysis["emotion"]:
            enhanced_query += f"\n- Emotion Detected: {analysis['emotion']}"
        if analysis["severity"]:
            enhanced_query += f"\n- Severity Level: {analysis['severity']}"
        if analysis["cause"]:
            enhanced_query += f"\n- Potential Cause: {', '.join(analysis['cause'])}"
        if analysis["effect"]:
            enhanced_query += f"\n- Resulting Effect: {', '.join(analysis['effect'])}"
            
        enhanced_query += "\n\nInstruction for AI: The user is seeking help. Use the provided context to answer their question. Write your response with deep empathy, acknowledging their current emotion and severity level."

        # ── E. Generate Answer with Gemma ──
        if summarizer:
            print("\n" + "=" * 70)
            print("  🤖 Answer:")
            print("=" * 70)
            print("  ", end="", flush=True)
            
            # Stream the response directly to the terminal!
            try:
                for chunk in summarizer.summarize(full_context_string, query=enhanced_query):
                    print(chunk, end="", flush=True)
                print() # Newline after response finishes
            except Exception as e:
                print(f"\n❌ Streaming error: {e}")
                
            print("\n" + "=" * 70)
        else:
            print("\n  ⚠️ Summarizer not loaded.")

if __name__ == "__main__":
    main()
