# Smart Contract Assistant

A modular, end-to-end **Retrieval-Augmented Generation (RAG)** web application for uploading and querying long documents (contracts, insurance policies, reports) via a conversational AI interface.

Built as part of the **NVIDIA DLI Course Workshop**, demonstrating LLM inference pipelines, LangChain LCEL, vector stores, semantic guardrails, and Gradio UI.

---

## Project Structure

```
smart_contract_assistant/
│
├── config.py           # Global configuration: models, prompts, chunking params
├── ingestion.py        # PDF/DOCX extraction, chunking, LCEL ingestion pipeline
├── vector_store.py     # FAISS vector store: build, save, load
├── rag_chain.py        # LCEL RAG chain, condense chain, summary chain
├── guardrails.py       # Pattern-based semantic guardrail (blocks harmful/chit-chat)
├── evaluation.py       # Batch evaluation pipeline with metrics
├── ui.py               # Gradio UI: Upload, Chat, Summarize, Evaluate tabs
│
├── main.py             # Entry point — launches Gradio UI
└── server.py           # FastAPI + LangServe microservice (REST API)
```

---

## Requirements

### Python Version

Python **3.9+** is required.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langserve[all]
fastapi
uvicorn
gradio
faiss-cpu
sentence-transformers
pdfplumber
python-docx
scikit-learn
numpy
python-dotenv
openai
```

---

## Environment Setup

**Never hardcode your API token.** Create a `.env` file in the project root:

```bash
# .env
HF_TOKEN=hf_your_token_here
```

Add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

Then update `config.py` to load from environment:

```python
import os
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", "")
```

Get your free HuggingFace token at: https://huggingface.co/settings/tokens

---

## Running the Application

### Option 1 — Gradio UI (Recommended for local use)

```bash
python main.py
```

Then open your browser at: **http://127.0.0.1:7860**

### Option 2 — FastAPI + LangServe REST API

```bash
python server.py
```

Then open: **http://localhost:8000/docs** for the Swagger UI.

> **Note:** The API requires a saved FAISS index. Upload a document via the Gradio UI first, then restart the server.

Available API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/rag/invoke` | RAG question answering |
| POST | `/summary/invoke` | Document summarization |
| GET | `/rag/playground` | LangServe interactive playground |

---

## How to Use

### Step 1 — Upload a Document

1. Open the **Upload & Process** tab
2. Click **Upload File** and select a `.pdf` or `.docx` file
3. Click **Process Document**
4. Wait for the success message showing chunk count

Alternatively, click **Load Saved Index** to restore a previously processed document.

### Step 2 — Ask Questions

1. Switch to the **Chat** tab
2. Type your question and press **Send** or hit **Enter**
3. The assistant answers using content from your document, with source citations showing which chunk each answer came from
4. Follow-up questions are supported — the system condenses conversation history automatically into a standalone query

### Step 3 — Summarize (Optional)

1. Go to the **Summarize** tab
2. Click **Generate 5-Point Summary**
3. The system returns a structured 5-bullet summary of the document's main topics

### Step 4 — Evaluate (Optional)

1. Go to the **Evaluate** tab
2. Enter one question per line, or use the default questions
3. Click **Run Evaluation**
4. Review the report showing answer previews, response times, faithfulness scores, and chunks used

---

## Architecture Overview

```
User Upload (PDF/DOCX)
        |
        v
  [Ingestion Pipeline]
  pdfplumber / python-docx
  RecursiveCharacterTextSplitter
  (chunk_size=1200, overlap=150)
        |
        v
  [Embedding]
  sentence-transformers/all-MiniLM-L6-v2
        |
        v
  [FAISS Vector Store]
  Saved to ./faiss_index/
        |
        v
  [LCEL RAG Chain]
  Retriever (top_k=5)
  -> format_docs
  -> ChatPromptTemplate (system + human)
  -> Llama-3.1-8B-Instruct via HuggingFace Router
  -> StrOutputParser
        |
        v
  [Guardrail Check]          [Condense Chain]
  Pattern-based filter   <-  (for follow-up questions)
        |
        v
  [Gradio UI / FastAPI]
  Answer + Source Citations
```

---

## Guardrails

The system uses a pattern-based guardrail (`guardrails.py`) that blocks two categories of queries:

- **Harmful content** — prompt injections, jailbreak attempts, requests for malware or weapon-making instructions
- **Chit-chat** — greetings, jokes, weather, cooking questions, sports scores, identity questions

Everything else is passed to the RAG chain. Off-topic questions (where the answer is not in the document) are handled gracefully by the LLM, which responds: *"The document does not contain information about [topic]."*

---

## Configuration Reference

All key parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | LLM used for Q&A |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `CHUNK_SIZE` | `1200` | Characters per document chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `700` | Max tokens in LLM response |
| `TEMPERATURE` | `0.2` | LLM temperature (lower = more factual) |
| `FAISS_DIR` | `faiss_index` | Directory for saved FAISS index |

---

## Evaluation

The built-in evaluation pipeline (`evaluation.py`) measures:

- **Response time** per question, in seconds
- **Grounding score** — word overlap between the answer and the retrieved source chunks
- **Chunks used** — confirms retrieval is working as configured

For best results, write document-specific questions rather than generic ones. Example for a networking textbook:

```
What is the difference between TCP and UDP?
What does the document say about socket programming?
How does DNS resolution work according to the document?
What transport layer protocols are described?
```

---

## Future Enhancements

- Multi-document search across a corpus
- Domain-specific fine-tuned embedding models
- Cloud deployment via Docker / Kubernetes
- Role-based access control
- Support for additional file formats (Excel, PowerPoint, HTML)
- Improved faithfulness metrics using an LLM-as-judge evaluation approach

---

