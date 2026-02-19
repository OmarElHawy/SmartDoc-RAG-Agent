
import gradio as gr
import pathlib
import os

from .ingestion import ingest_document
from .vector_store import build_vector_store, load_vector_store
from .rag_chain import build_rag_chain, build_summary_chain, run_rag, run_summary
from .evaluation import evaluate
from ..config import QA_SYSTEM_PROMPT

DEFAULT_EVAL = """What are the main topics covered?
Key concepts explained?
Conclusions mentioned?
Main definitions?"""

_vector_store   = None
_rag_chain      = None
_condense_chain = None
_retriever      = None
_summary_chain  = None
_all_docs       = []
_chat_history_list = [] 
_chat_history_str  = "" 

def ui_upload(file):
    global _vector_store, _rag_chain, _condense_chain, _retriever, _summary_chain, _all_docs, _chat_history_list, _chat_history_str
    
    if file is None:
        return "❌ Please upload a PDF or DOCX file."
    
    try:
        _all_docs = ingest_document(file.name)
        _vector_store = build_vector_store(_all_docs)
        _rag_chain, _condense_chain, _retriever = build_rag_chain(_vector_store)
        _summary_chain = build_summary_chain()
        
        _chat_history_list = []
        _chat_history_str = ""
        
        return (
            f"✅ **Document processed!**\n\n"
            f"- File: `{pathlib.Path(file.name).name}`\n"
            f"- Chunks: `{len(_all_docs)}`\n"
            f"- LCEL Chain: `retriever | format_docs | prompt | llm | StrOutputParser`\n\n"
            f"Go to **Chat** to ask questions!"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ui_load_index():
    global _vector_store, _rag_chain, _condense_chain, _retriever, _summary_chain, _chat_history_list, _chat_history_str
    try:
        _vector_store = load_vector_store()
        _rag_chain, _condense_chain, _retriever = build_rag_chain(_vector_store)
        _summary_chain = build_summary_chain()
        
        # Reset chat
        _chat_history_list = []
        _chat_history_str = ""
        
        return "✅ Loaded index. LCEL RAG chain ready!"
    except Exception as e:
        return f"❌ {str(e)}"

def ui_chat(user_message, history):
    global _rag_chain, _condense_chain, _retriever, _chat_history_str, _chat_history_list
    
    if not user_message.strip():
        return "", history

    if history is None:
        history = []


    if _rag_chain is None:
        return "", history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": "⚠️ No document loaded. Please upload first."}]

    try:
        answer, sources, _ = run_rag(_rag_chain, _condense_chain, _retriever, user_message, _chat_history_str)
        
        seen, src_lines = set(), []
        for doc in sources:
            src = doc.metadata.get("source", "?")
            idx = doc.metadata.get("chunk_index", "?")
            key = f"{src}#{idx}"
            if key not in seen:
                seen.add(key)
                preview = doc.page_content[:80].replace("\n", " ")
                src_lines.append(f"📄 {src} (chunk {idx}): {preview}...")

        full_response = str(answer)
        if src_lines:
            full_response += "\n\n**Sources**:\n" + "\n".join(src_lines)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": full_response})
        _chat_history_str += f"\nHuman: {user_message}\nAssistant: {full_response}"
        
        return "", history

    except Exception as e:
        return "", history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": f"❌ Error: {str(e)}"}]

def ui_clear():
    global _chat_history_list, _chat_history_str
    _chat_history_list = []
    _chat_history_str = ""
    return []

def ui_summarize():
    if not _all_docs or _summary_chain is None:
        return "⚠️ No document loaded."
    try:
        summary = run_summary(_summary_chain, _all_docs)
        return f"## Summary\n\n{summary}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def ui_evaluate(questions_text):
    if _rag_chain is None:
        return "⚠️ No document loaded."
    questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
    if not questions:
        return "⚠️ Enter at least one question."
    try:
        from .rag_chain import run_rag 
        return evaluate(questions, _rag_chain, _condense_chain, _retriever)
    except Exception as e:
        return f"❌ Error: {str(e)}"


def build_app():
    with gr.Blocks(title="Smart Contract Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# 📜 Smart Contract Assistant
**Modular Architecture** | **LCEL Pipeline** | **RAG & Summarization**
""")
        
        with gr.Tab("📁 Upload & Process"):
            gr.Markdown("### Upload your PDF or DOCX document")
            file_input    = gr.File(label="Upload File", file_types=[".pdf", ".docx"])
            with gr.Row():
                upload_btn    = gr.Button("⚙️ Process Document", variant="primary")
                load_btn      = gr.Button("📂 Load Saved Index",  variant="secondary")
            upload_status = gr.Markdown("*No document loaded yet.*")
            
            upload_btn.click(ui_upload,   inputs=file_input, outputs=upload_status)
            load_btn.click(ui_load_index, inputs=None,       outputs=upload_status)



        with gr.Tab("💬 Chat"):
            gr.Markdown("### Conversational Q&A with Citations")
            chatbot   = gr.Chatbot(label="Chat History", height=500)
            msg_input = gr.Textbox(placeholder="Ask about the document...", label="Your Question")
            with gr.Row():
                send_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            send_btn.click(ui_chat,  inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(ui_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            clear_btn.click(ui_clear, outputs=chatbot)

        with gr.Tab("📋 Summarize"):
            gr.Markdown("### Generate Executive Summary")
            sum_btn = gr.Button("✨ Generate 5-Point Summary", variant="primary")
            sum_out = gr.Markdown("*Click button to generate.*")
            sum_btn.click(ui_summarize, outputs=sum_out)

        with gr.Tab("🧪 Evaluate"):
            gr.Markdown("### Batch Evaluation Pipeline")
            eval_in  = gr.Textbox(value=DEFAULT_EVAL, label="Test Questions", lines=6)
            eval_btn = gr.Button("▶️ Run Evaluation", variant="primary")
            eval_out = gr.Textbox(label="Evaluation Report", lines=20, interactive=False)
            eval_btn.click(ui_evaluate, inputs=eval_in, outputs=eval_out)

    return demo
