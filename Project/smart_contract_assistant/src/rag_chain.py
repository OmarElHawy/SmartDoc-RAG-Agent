
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models import ChatOpenAI
from ..config import (
    LLM_MODEL, HF_TOKEN, HF_BASE_URL, MAX_TOKENS, TEMPERATURE, TOP_K,
    QA_SYSTEM_PROMPT, CONDENSE_TEMPLATE, SUMMARY_SYSTEM_PROMPT
)

os.environ["HF_TOKEN"]        = HF_TOKEN
os.environ["OPENAI_API_KEY"]  = HF_TOKEN
os.environ["OPENAI_API_BASE"] = HF_BASE_URL

def get_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=HF_TOKEN,
        openai_api_base=HF_BASE_URL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

def format_docs(docs: list) -> str:
    """Format retrieved docs into a single context string."""
    return "\n\n---\n\n".join(
        f"[Chunk {d.metadata.get('chunk_index','?')} | {d.metadata.get('source','?')}]\n{d.page_content}"
        for d in docs
    )

def build_rag_chain(vector_store):
    """
    LCEL RAG Chain:
    {context: retriever | format_docs, question: passthrough}
    | QA_PROMPT | llm | StrOutputParser
    """
    llm       = get_llm()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    condense_prompt = PromptTemplate.from_template(CONDENSE_TEMPLATE)

    rag_chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    condense_chain = (
        condense_prompt
        | llm
        | StrOutputParser()
    )

    print("[RAG] LCEL chain built: retriever | format_docs | prompt | llm | StrOutputParser")
    return rag_chain, condense_chain, retriever

def build_summary_chain():
    """
    LCEL Summary Chain:
    SUMMARY_PROMPT | llm | StrOutputParser
    """
    llm = get_llm()
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM_PROMPT),
        ("human", "Document content:\n\n{text}"),
    ])

    summary_chain = (
        summary_prompt
        | llm
        | StrOutputParser()
    )

    print("[Summary] LCEL chain built: SUMMARY_PROMPT | llm | StrOutputParser")
    return summary_chain

def run_rag(rag_chain, condense_chain, retriever, question: str, chat_history_str: str) -> tuple:

    if chat_history_str.strip():
        standalone = condense_chain.invoke({
            "chat_history": chat_history_str,
            "question": question,
        })
    else:
        standalone = question

    from .guardrails import get_guardrail
    guard = get_guardrail()
    check_result = guard.check(standalone)
    
    if not check_result["is_allowed"]:
        return (
            f"🚫 **Guardrail Blocked**: I cannot answer this query because it seems off-topic or irrelevant to smart contracts. (Confidence: {check_result['score']:.2f})",
            [], 
            standalone
        )

    source_docs = retriever.invoke(standalone)

    answer = rag_chain.invoke(standalone)

    return answer, source_docs, standalone

def run_summary(summary_chain, docs: list) -> str:
    combined = "\n\n".join(d.page_content for d in docs[:6])
    return summary_chain.invoke({"text": combined})
