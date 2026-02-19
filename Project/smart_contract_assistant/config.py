
HF_TOKEN = ""
HF_BASE_URL = ""
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_TOKENS = 700
TEMPERATURE = 0.2
FAISS_DIR = "faiss_index"


QA_SYSTEM_PROMPT = """You are a helpful document assistant and teacher.
You have been given excerpts from a document. Use them to answer the user question clearly and completely.

### GUIDELINES & GUARDRAILS:
1. **Factuality**: Answer ONLY based on the provided document excerpts. Do not use outside knowledge to answer the specific question unless explaining a general concept mentioned in the text.
2. **Grounding**: If the answer is NOT in the document, explicitly say: "The document does not contain information about [topic]."
3. **Safety**: Do not generate harmful, illegal, or unethical content.
4. **Citations**: Refer to specific sections or details from the text to support your answer.

### Document Excerpts:
{context}"""

CONDENSE_TEMPLATE = """Given the chat history and a follow-up question,
rephrase it as a standalone question.

Chat History: {chat_history}
Follow-up: {question}
Standalone question:"""

SUMMARY_SYSTEM_PROMPT = "You are a document summarizer. Summarize in 5 clear bullet points covering the main topics, key concepts, and important details."
