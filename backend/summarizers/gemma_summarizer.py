"""
Gemma Summarizer (via Ollama)
=============================

Connects to the local Ollama instance running on your Mac.
This is lightning fast because Ollama handles the quantization and Metal GPU acceleration!
"""

import json
import urllib.request
import urllib.error

# The exact name of the model you downloaded in Ollama. 
# Typical names are "gemma", "gemma:2b", or "gemma:7b". 
OLLAMA_MODEL_NAME = "gemma3:4b" 


class GemmaSummarizer:
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME):
        self.model_name = model_name
        self._check_ollama()

    def _check_ollama(self):
        print(f"\n{'='*60}")
        print(f"  Connecting to local Ollama API")
        print(f"  Model: {self.model_name}")
        print(f"{'='*60}")
        
        try:
            # Simple check to see if Ollama is running
            req = urllib.request.Request("http://localhost:11434/")
            urllib.request.urlopen(req, timeout=2)
            print("  ✅ Successfully connected to Ollama!")
            print(f"{'='*60}\n")
        except urllib.error.URLError:
            print("  ❌ Could not connect to Ollama.")
            print("  ⚠️ Make sure the Ollama app is open and running on your Mac!")
            print(f"{'='*60}\n")
            self.model_name = None

    def summarize(self, context_text: str, query: str = "") -> str:
        if not self.model_name:
            return "❌ Ollama is not running."

        # Instruct prompt format
        prompt = (
            "You are an expert mental health and medical AI assistant. "
            "Use ONLY the following provided context from medical textbooks and Reddit discussions "
            "to answer the user's question. Do NOT add any information that is not present in the context. "
            "Do NOT hallucinate or fabricate facts. If the context does not contain enough information "
            "to fully answer the question, clearly state what is unknown rather than guessing. "
            "Your response MUST be between 200 and 300 tokens — detailed yet concise.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        print("\n  [Summarizer] Asking Ollama for the answer...")

        try:
            url = "http://localhost:11434/api/generate"
            data = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": 400  # upper bound in tokens to allow ~200-300 token answers
                }
            }).encode('utf-8')

            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            # Timeout set to 60 seconds so it doesn't freeze forever
            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            yield chunk["response"]

        except urllib.error.URLError as e:
            yield f"❌ Error connecting to Ollama: {e.reason}"
        except TimeoutError:
            yield "\n\n⚠️ Ollama took too long to respond (timeout). It might be downloading the model or struggling to process the context."
        except Exception as e:
            yield f"❌ Error during generation: {e}"

    def generate_chitchat(self, query: str) -> str:
        if not self.model_name:
            yield "❌ Ollama is not running."
            return

        prompt = (
            "You are a friendly, empathetic mental health AI assistant. "
            "The user is just having a casual conversation or saying hello. "
            "Respond warmly, politely, and concisely.\n\n"
            f"User: {query}\n\n"
            "Assistant:"
        )

        print("\n  [Summarizer] Generating quick response (chitchat)...")

        try:
            url = "http://localhost:11434/api/generate"
            data = json.dumps({
                "model": self.model_name,
                "prompt": prompt,
                "stream": True
            }).encode('utf-8')

            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if "response" in chunk:
                            yield chunk["response"]

        except urllib.error.URLError as e:
            yield f"❌ Error connecting to Ollama: {e.reason}"
        except TimeoutError:
            yield "\n\n⚠️ Ollama took too long to respond."
        except Exception as e:
            yield f"❌ Error during generation: {e}"
