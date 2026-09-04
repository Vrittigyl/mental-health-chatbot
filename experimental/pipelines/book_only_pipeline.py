import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to sys.path for cross-folder imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from backend.retrievers.book_retriever import HybridRetriever, clean_output_text
from backend.summarizers.gemma_summarizer import GemmaSummarizer
from backend.pipeline import analyze_query
from backend.metrics.router import route_query

def main():
    print("\n" + "=" * 70)
    print("  📚 Mental Health Chatbot — Book-Only Pipeline")
    print("  (Emotion/Cause/Severity + Textbooks + Gemma LLM)")
    print("=" * 70)

    # ── Step 1: Load Knowledge Retriever (Textbooks) ──
    print("\n📚 Loading Knowledge Retriever (Textbooks)...")
    try:
        book_retriever = HybridRetriever()
        print("✅ Textbook Retriever loaded!\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # ── Step 2: Load Summarizer (Gemma via Ollama) ──
    try:
        summarizer = GemmaSummarizer()
    except Exception as e:
        print(f"❌ Could not load Gemma summarizer: {e}")
        summarizer = None

    # ── Step 3: Interactive Chat Loop ──
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

        if not user_input:
            print("\nGoodbye! 👋")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break

        # ── 0. Route Query ──
        print("\n🚦 STEP 0: Routing User Query...")
        route_category = route_query(user_input)
        print(f"  Route Category: {route_category}")
        
        if route_category == "crisis":
            from backend.metrics.router import CRISIS_RESPONSE
            print("\n🚨 CRISIS DETECTED — Bypassing pipeline.")
            print(f"\n  🤖 Assistant: {CRISIS_RESPONSE}")
            print("\n")
            continue
        elif route_category == "unrelated":
            print("\n  🤖 Assistant: I am a mental health assistant and I can only answer questions related to mental health.")
            continue
        elif route_category == "greeting":
            print("\n👋 Greeting detected! Skipping retrieval and calling chitchat.")
            print("\n  🤖 Assistant:", end=" ", flush=True)
            if summarizer:
                for chunk in summarizer.generate_chitchat(user_input):
                    print(chunk, end="", flush=True)
            else:
                print("Hello there! How can I help you today?")
            print("\n")
            continue

        # ── A. Analyze the User's Query ──
        analysis = analyze_query(user_input)
        
        # ── B. Retrieve from Textbooks ONLY ──
        combined_contexts = []
        print("\n  🔍 Searching Textbooks...")
        book_results = book_retriever.retrieve_top_k(user_input, k=5, similarity_pool=15)
        
        if book_results:
            combined_contexts.append("=== MEDICAL TEXTBOOK EXCERPTS ===")
            print(f"  📖 Found {len(book_results)} textbook results:")
            for i, result in enumerate(book_results, 1):
                cleaned = clean_output_text(result["text"])
                print(f"     [{i}] [{result['source']}] (Score: {result['score']:.4f}) {cleaned}")
                combined_contexts.append(f"[Textbook: {result['source']}]\n{cleaned}\n")
        else:
            print("  📖 No textbook results found.")

        # ── C. Prepare Prompt for Gemma ──
        if not combined_contexts:
            print("\n  🤖 Answer: I couldn't find any relevant information to answer that.")
            continue

        full_context_string = "\n".join(combined_contexts)

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

        # ── D. Generate Answer with Gemma ──
        if summarizer:
            print("\n" + "=" * 70)
            print("  🤖 Answer:")
            print("=" * 70)
            print("  ", end="", flush=True)
            
            try:
                for chunk in summarizer.summarize(full_context_string, query=enhanced_query):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n❌ Streaming error: {e}")
                
            print("\n" + "=" * 70)
        else:
            print("\n  ⚠️ Summarizer not loaded.")

if __name__ == "__main__":
    main()
