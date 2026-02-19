
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from smart_contract_assistant.src.rag_chain import build_rag_chain, build_summary_chain
from smart_contract_assistant.src.vector_store import load_vector_store
from smart_contract_assistant.config import HF_TOKEN

app = FastAPI(
    title="Smart Contract Assistant API",
    version="1.0",
    description="A LangServe API for the Smart Contract Assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None
summary_chain = None

@app.on_event("startup")
async def startup_event():
    global rag_chain, summary_chain
    try:
        print("[Server] Loading vector store...")
        vector_store = load_vector_store()
        
        built_rag, _, _ = build_rag_chain(vector_store)
        rag_chain = built_rag
        
        summary_chain = build_summary_chain()
        print("[Server] Chains loaded successfully.")
    except Exception as e:
        print(f"[Server] Warning: Could not load index on startup. {e}")
        print("[Server] Please create an index first using the Gradio UI.")

@app.get("/")
async def redirect_root_to_docs():
    return {"message": "Welcome to Smart Contract Assistant API. Go to /docs for API docs or /rag/playground for RAG."}


try:
    if os.path.exists("faiss_index"):
        vs = load_vector_store()
        r_chain, _, _ = build_rag_chain(vs)
        s_chain = build_summary_chain()
        
        add_routes(app, r_chain, path="/rag")
        add_routes(app, s_chain, path="/summary")
    else:
        print("[Server] 'faiss_index' not found. Routes /rag and /summary will not be available until restart after data ingestion.")
except Exception as e:
    print(f"[Server] Error setting up routes: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
