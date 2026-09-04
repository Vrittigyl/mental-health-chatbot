# 🧠 MindMend: Mental Health AI Chatbot

Welcome to the **MindMend** project! This is an advanced, privacy-first Mental Health AI Chatbot. It runs entirely locally on your machine, leveraging state-of-the-art NLP models for dialog analysis, a dual-RAG (Retrieval-Augmented Generation) system reading from both clinical textbooks and Reddit discussions, and the Gemma LLM via Ollama to generate highly empathetic, context-aware advice.

---

## 📂 Project Structure

```
mental-health-chatbot/
├── frontend/           ← React + Vite frontend UI
├── backend/            ← Core RAG pipeline & API server
└── experimental/       ← Experimental scripts & alternative pipelines
```

### `frontend/`
The React/TypeScript web UI built with Vite. Connects to the backend API to provide a chat interface.

### `backend/`
The core application logic — everything needed to run the chatbot:
- **`api.py`** — FastAPI server (entry point for frontend)
- **`final_pipeline.py`** — CLI-based interactive pipeline (terminal entry point)
- **`pipeline.py`** — NLP query analysis orchestrator (emotion, severity, intent, cause)
- **`config/`** — Centralized settings and hyperparameters
- **`data/`** — Textbook paragraphs, Reddit Q&A, embeddings
- **`metrics/`** — NLP classification models (emotion, severity, intent, cause)
- **`retrievers/`** — RAG search engines (textbooks, Reddit, knowledge graph)
- **`summarizers/`** — Gemma LLM summarizer (via Ollama)
- **`requirements.txt`** — Python dependencies

### `experimental/`
Scripts and alternative pipeline variants used for experimentation, **not part of the main architecture**:
- **`pipelines/`** — Alternative pipelines (BART, BigBird, Pegasus-X, Gemma, Book-only, Reddit-only)
- **`summarizers/`** — Alternative summarizer implementations (BART, BigBird, Pegasus-X, T5)
- **`dataset_preparation/`** — One-time data preparation scripts
- **`preprocessing/`** — Reddit data preprocessing scripts
- **`bert_embedding.py`** — SBERT Q&A embedding experiment
- **`keyword_extraction_test.py`** — KeyBERT/RAKE keyword extraction test
- **`sbert_keyphrase_pipeline.py`** — SBERT keyphrase similarity pipeline

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Query]) --> FinalPipeline[final_pipeline.py<br>Master Orchestrator]
    
    FinalPipeline --> NLP
    FinalPipeline --> Retrieval
    
    subgraph NLP [1. NLP Dialog Analysis]
    direction TB
        E[Emotion Detector<br>roberta-base]
        S[Severity Detector<br>deberta-v3-large]
        I[Intent Detector<br>bart-large-mnli]
        C[Cause/Effect Extractor<br>unicausal-tok]
    end
    
    subgraph Retrieval [2. RAG Knowledge Retrieval]
    direction TB
        B[Textbook Retriever<br>Semantic + MMR Search]
        R[Reddit Retriever<br>Cosine Similarity / FAISS]
    end
    
    NLP --> Merge{Context Merger}
    Retrieval --> Merge
    
    Merge --> LLM[Ollama Local API<br>Gemma3:4b]
    LLM --> Stream([Streaming Response Back to User])
```

### How It Works:
1. **User Query**: The user types a message (via terminal or web UI).
2. **Dialog Analysis (`backend/pipeline.py`)**: Analyzes the text for emotion, severity, intent, and root causes/effects.
3. **Retrieval (`backend/retrievers/`)**: 
   - **Textbooks**: Searches chunks of 5 major psychology textbooks using semantic embeddings.
   - **Reddit**: Searches 20,000+ real-world Reddit discussions from mental health subreddits.
4. **LLM Generation (`backend/summarizers/gemma_summarizer.py`)**: All structured data is bundled into a prompt and sent to a local LLM, which streams back a compassionate, clinically-informed response.

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Install Prerequisites
1. **Python 3.10+**
2. **Node.js 18+** (for the frontend)
3. **Ollama**: Download and install [Ollama](https://ollama.com/). Keep it running in the background.

### Step 2: Set Up Python Environment
```bash
cd backend
python3 -m venv ../venv
source ../venv/bin/activate  # On Windows: ..\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Download the Required Data
Because the raw data files are too large for GitHub, you must download and place them in the correct folders:
1. **Textbooks**: Place clinical psychology PDF books inside `backend/data/books/`
2. **Reddit Data**: Place Reddit CSV files inside `backend/data/reddit/`

### Step 4: Download the Local LLM (Ollama)
```bash
ollama pull gemma3:4b
```
*(⚠️ Download Size: ~3.3GB)*

### Step 5: Build the Knowledge Graphs
Generate dense vector embeddings locally (run once):
```bash
cd backend
python3 -m retrievers.knowledge_graph.ingest
python3 -m retrievers.knowledge_graph.ingest_embeddings
```

---

## 💻 Running the Chatbot

### Terminal Mode (CLI)
```bash
cd backend
python3 final_pipeline.py
```

### API Server Mode (for Frontend)
```bash
cd backend
uvicorn api:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### What happens on the FIRST run?
Hugging Face will automatically download 4 AI models (~5GB total):
- **Emotion Model** (`roberta-base`)
- **Severity Model** (`deberta-v3-large`)
- **Intent Model** (`bart-large-mnli`)
- **Cause Extractor** (`unicausal-tok`)

*(Subsequent runs will load instantly from cache!)*

Type `quit` at any time to exit the terminal chatbot.

---

Happy building! 🧠💙