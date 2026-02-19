
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import EMBEDDING_MODEL, FAISS_DIR

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print(f"[Embeddings] Loading {EMBEDDING_MODEL} ...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embeddings] Loaded.")
    return _embeddings

def build_vector_store(docs: list) -> FAISS:
    print(f"[VectorStore] Embedding {len(docs)} chunks...")
    vs = FAISS.from_documents(docs, get_embeddings())
    vs.save_local(FAISS_DIR)
    print(f"[VectorStore] Saved to '{FAISS_DIR}'.")
    return vs

def load_vector_store() -> FAISS:
    if not os.path.exists(FAISS_DIR):
        raise FileNotFoundError("No saved index. Please upload a document first.")
    vs = FAISS.load_local(FAISS_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    print(f"[VectorStore] Loaded from '{FAISS_DIR}'.")
    return vs
