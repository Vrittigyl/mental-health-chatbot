import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from retrievers.book_retriever import HybridRetriever, clean_output_text
from retrievers.reddit_retriever import load_training_data, find_similar
from summarizers.gemma_summarizer import GemmaSummarizer
from pipeline import analyze_query

# Global objects for models
book_retriever = None
reddit_data = []
summarizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global book_retriever, reddit_data, summarizer
    print("🚀 Starting up API Server... Loading models into memory.")
    
    # 1. Load Textbook Retriever
    print("\n📚 Loading Knowledge Retriever (Textbooks)...")
    try:
        book_retriever = HybridRetriever()
        print("✅ Textbook Retriever loaded!")
    except FileNotFoundError as e:
        print(f"❌ Error loading Textbooks: {e}")

    # 2. Load Reddit Data
    print("\n💬 Loading Reddit Q&A Dataset...")
    try:
        reddit_data = load_training_data()
        print(f"✅ Reddit Dataset loaded! ({len(reddit_data)} items)")
    except Exception as e:
        print(f"❌ Error loading Reddit data: {e}")

    # 3. Load Gemma Summarizer
    print("\n🤖 Loading Gemma Summarizer...")
    try:
        summarizer = GemmaSummarizer()
        print("✅ Gemma Summarizer loaded!")
    except Exception as e:
        print(f"❌ Could not load Gemma summarizer: {e}")
        
    print("\n✨ API Server is ready!")
    yield
    print("🛑 Shutting down API Server...")

app = FastAPI(title="Mental Health Chatbot API", lifespan=lifespan)

# Allow your frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.query.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    print("\n" + "=" * 70)
    print(f"  📨 NEW REQUEST: \"{user_input}\"")
    print("=" * 70)

    # ── 1. Analyze Query ──
    print("\n📊 STEP 1: Analyzing User Query...")
    analysis = analyze_query(user_input)

    print("\n" + "-" * 50)
    print("  📊 ANALYSIS RESULTS:")
    print(f"     Emotion  : {analysis.get('emotion', 'N/A')}")
    print(f"     Severity : {analysis.get('severity', 'N/A')}")
    print(f"     Cause    : {', '.join(analysis.get('cause', [])) or 'None detected'}")
    print(f"     Effect   : {', '.join(analysis.get('effect', [])) or 'None detected'}")
    print(f"     Signal   : {', '.join(analysis.get('signal', [])) or 'None detected'}")
    intent = analysis.get('intent', {})
    if intent.get('action') and intent.get('object'):
        print(f"     Intent   : Action='{intent['action']}', Object='{intent['object']}'")
    else:
        print(f"     Intent   : None detected")
    print("-" * 50)

    # ── 2. Retrieve Textbooks ──
    print("\n📚 STEP 2: Searching Textbooks...")
    combined_contexts = []
    if book_retriever:
        book_results = book_retriever.retrieve_top_k(user_input, k=3, similarity_pool=10)
        if book_results:
            combined_contexts.append("=== MEDICAL TEXTBOOK EXCERPTS ===")
            print(f"  ✅ Found {len(book_results)} textbook paragraphs:")
            for i, result in enumerate(book_results, 1):
                cleaned = clean_output_text(result["text"])
                combined_contexts.append(f"[Textbook: {result['source']}]\n{cleaned}\n")
                # Print a preview of each textbook result
                preview = cleaned[:200] + "..." if len(cleaned) > 200 else cleaned
                print(f"\n  📖 [{i}] Source: {result['source']}")
                print(f"       Preview: {preview}")
        else:
            print("  ⚠️ No relevant textbook paragraphs found.")
    else:
        print("  ❌ Book retriever not loaded.")

    # ── 3. Retrieve Reddit ──
    print("\n💬 STEP 3: Searching Reddit Discussions...")
    if reddit_data and book_retriever:
        query_emb = book_retriever.model.encode(user_input)
        reddit_results = find_similar(query_emb, reddit_data, top_k=3)
        if reddit_results:
            combined_contexts.append("=== REDDIT DISCUSSIONS ===")
            print(f"  ✅ Found {len(reddit_results)} Reddit discussions:")
            for i, result in enumerate(reddit_results, 1):
                q = result["question"]
                ans = result["answers"][0]["answer"] if result["answers"] else "No answer available."
                if len(ans) > 500:
                    ans = ans[:500] + "..."
                combined_contexts.append(f"[Reddit Discussion - {result['disease']}]\nQuestion: {q}\nTop Answer: {ans}\n")
                # Print a preview of each Reddit result
                ans_preview = ans[:150] + "..." if len(ans) > 150 else ans
                print(f"\n  💬 [{i}] Disease: {result['disease']}")
                print(f"       Question: {q[:100]}{'...' if len(q) > 100 else ''}")
                print(f"       Answer:   {ans_preview}")
        else:
            print("  ⚠️ No relevant Reddit discussions found.")
    else:
        print("  ⚠️ Reddit data or book retriever not available.")

    # ── 4. Prompt for LLM ──
    if not combined_contexts:
        print("\n  ❌ No context found. Returning fallback answer.")
        return {
            "analysis": analysis,
            "answer": "I couldn't find any relevant medical information to answer that."
        }

    full_context_string = "\n".join(combined_contexts)

    enhanced_query = f"{user_input}\n\n[User's Current Mental State Profile]"
    if analysis.get("emotion"):
        enhanced_query += f"\n- Emotion Detected: {analysis['emotion']}"
    if analysis.get("severity"):
        enhanced_query += f"\n- Severity Level: {analysis['severity']}"
    if analysis.get("cause"):
        enhanced_query += f"\n- Potential Cause: {', '.join(analysis['cause'])}"
    if analysis.get("effect"):
        enhanced_query += f"\n- Resulting Effect: {', '.join(analysis['effect'])}"
        
    enhanced_query += "\n\nInstruction for AI: The user is seeking help. Use the provided context to answer their question. Write your response with deep empathy, acknowledging their current emotion and severity level."

    print("\n" + "-" * 50)
    print("  🧠 ENHANCED QUERY SENT TO LLM:")
    print("-" * 50)
    print(enhanced_query)
    print("-" * 50)

    # ── 5. Generate Answer ──
    print("\n🤖 STEP 4: Generating Answer with Gemma LLM...")
    final_answer = ""
    if summarizer:
        try:
            for chunk in summarizer.summarize(full_context_string, query=enhanced_query):
                final_answer += chunk
        except Exception as e:
            final_answer += f"\n[Error generating full response: {e}]"
    else:
        final_answer = "Summarizer is not available."

    print("\n" + "=" * 70)
    print("  ✅ FINAL ANSWER GENERATED:")
    print("=" * 70)
    print(final_answer.strip())
    print("=" * 70 + "\n")

    return {
        "analysis": analysis,
        "answer": final_answer.strip()
    }

