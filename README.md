# AI Diagnosis Assistant — README

**AI Diagnosis Assistant** is a retrieval-augmented generation (RAG) prototype that answers medical questions using a knowledge base built from *Harrison’s Principles of Internal Medicine*. It combines FAISS vector search, SentenceTransformers embeddings, and an LLM (Google Gemini by default) behind a Streamlit UI. This project is an **educational prototype only** — not a substitute for professional medical advice.

---

## Features
- Build a vectorized knowledge base (PDF → text chunks → embeddings → FAISS).
- Retrieval of relevant sections from the knowledge base.
- RAG-style prompt assembly and LLM reasoning (Gemini / HuggingFace / local fallback).
- Streamlit UI for symptom input + PDF upload.
- Source page citation and transparent disclaimer.

---

# Snapshot

<img width="764" height="595" alt="image" src="https://github.com/user-attachments/assets/cc29966c-aeea-43ca-97ce-7bc4253ffc8b" />
<img width="764" height="521" alt="image" src="https://github.com/user-attachments/assets/6402a2b0-962b-4060-a454-252996fce732" />
<img width="748" height="538" alt="image" src="https://github.com/user-attachments/assets/f3bc66b7-0434-45b9-99a1-0b97f9a1f8a5" />

---

## Repo layout (important files)
- `create_knowledge_base.py` — extract PDF, chunk text, embed, create & save FAISS store.
- `diagnosis_assistant.py` — Streamlit app: loads vectorstore, retrieves docs, formats prompt, calls LLM, displays answer.
- `harrison_vectorstore/` — (generated) FAISS vectorstore directory (created by `create_knowledge_base.py`).
- `.env` — (local, not tracked) holds `GEMINI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN` (optional).
- `requirements.txt` — suggested dependencies (see below).

---

## Quick start (local)

### 1. Clone repo
```bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv venv
# activate the virtualenv:
# Windows PowerShell:
venv\Scripts\Activate.ps1
# or cmd:
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate
```

### 2. Install dependencies
Create a `requirements.txt` (example below) and install:
```bash
pip install -r requirements.txt
```

**Suggested `requirements.txt` (example pinned versions)**  
```
streamlit>=1.20.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.1
sentence-transformers>=2.2.2
transformers>=4.33.0
faiss-cpu>=1.7.4
google-generativeai>=0.3.0
python-dotenv>=1.0.0
PyMuPDF>=1.22.0
tqdm>=4.64.0
```
> You may need to adjust FAISS package for your platform (e.g., `faiss-gpu`) or install from conda if wheel not available.

### 3. Add environment keys
Create a `.env` file in project root:

```
# .env
GEMINI_API_KEY=your_gemini_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here   # optional
```

Load with `python-dotenv` (the app already calls `load_dotenv()`).

### 4. Build the knowledge base (run once)
Edit `create_knowledge_base.py` to point to your Harrison PDF path (variable already set in the script). Then run:

```bash
python create_knowledge_base.py
```

This will:
- Extract text pages from the PDF
- Chunk the text into smaller documents
- Create embeddings and build a FAISS vectorstore at `harrison_vectorstore/`

### 5. Run the Streamlit app
```bash
streamlit run diagnosis_assistant.py
```
Open the URL Streamlit provides (usually `http://localhost:8501`).

---

## How the RAG flow works (high level)
1. User provides symptoms and/or uploads a PDF.
2. App formats a `question` or `query`.
3. App uses FAISS retriever to get top-k relevant chunks from the Harrison knowledge base.
4. The context chunks are concatenated and combined with the user question into a prompt (PromptTemplate).
5. LLM (Gemini preferred) is called with the prompt and returns a context-grounded answer.
6. The UI displays the answer + source page metadata and a disclaimer.

---

## LLM options & fallbacks
The app chooses the LLM in this order:

1. **HuggingFace Inference API** — used if `HUGGINGFACEHUB_API_TOKEN` set and the `HuggingFaceEndpoint` is enabled.
2. **Google Gemini** — used if `GEMINI_API_KEY` set (via `google.generativeai`).
3. **Local model** — a small local text-generation pipeline (e.g., `distilbert`) used as a last resort — for demo only (poor quality).

> Gemini is preferred for reasoning and multi-turn tasks; HuggingFace is a good alternative if you prefer an HF model.

---

## Common troubleshooting & gotchas

### `ImportError: cannot import name 'Embeddings'`
LangChain moves APIs between versions. Use:
```py
from langchain.embeddings.base import Embeddings
```
and subclass this for a stable interface.

### `'<YourEmbeddingsClass>' object is not callable`
FAISS expects an **Embeddings object** implementing the LangChain `Embeddings` interface (methods `embed_documents` and `embed_query`). Make your class inherit `Embeddings` and return python lists (not numpy arrays).

### `Missing some input keys: {'query'}` or `question` mismatch
LangChain versions differ about input keys:
- Newer versions use `"question"` by default.
- Older versions used `"query"`.
To be version-robust, either:
- Avoid `RetrievalQA.from_chain_type(...)` and implement RAG manually: retrieve docs → format prompt → call LLM (recommended if you encounter key mismatches).
- Or explicitly match the key your chain expects when calling (or set `input_key` where supported).

The project contains a manual RAG implementation (no `RetrievalQA`) to avoid this mismatch.

### `Chain.__call__ deprecation` warning
Use `.invoke()` where supported.

### FAISS installation issues
If `pip install faiss-cpu` fails on Windows, consider using conda or prebuilt wheels.

---

## Safety & legal
- This project is an **educational demo** — not a medical device.
- Always display a prominent disclaimer in any UI: **“This is not a substitute for professional medical advice.”**
- Do **not** deploy with patient-identifiable data without proper legal & security safeguards.

---

## Recommended improvements (future)
- Add authentication & rate-limiting.
- Improve prompts (structured JSON outputs).
- Replace weak local fallback with stronger OSS LLMs (e.g., Mistral, Llama).
- Containerize with Docker.

---

## License & credits
- Code is provided without warranty. Use responsibly.
- Libraries: LangChain, SentenceTransformers, FAISS, Google Generative AI, Streamlit.

