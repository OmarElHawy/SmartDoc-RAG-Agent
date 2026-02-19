
import time

def evaluate(questions: list, rag_chain, condense_chain, retriever) -> str:
    lines      = ["=" * 55, "  EVALUATION REPORT", "=" * 55]
    total_time = 0

    chat_history_str = ""

    from .rag_chain import run_rag

    for i, question in enumerate(questions, 1):
        start   = time.time()
        answer, sources, _ = run_rag(rag_chain, condense_chain, retriever, question, chat_history_str)
        elapsed = round(time.time() - start, 2)
        total_time += elapsed

        src_words = set(" ".join(d.page_content for d in sources).lower().split())
        ans_words = set(answer.content.lower().split()) if hasattr(answer, 'content') else set(str(answer).lower().split())
        
        faith     = round(len(ans_words & src_words) / max(len(ans_words), 1), 2)

        lines.append(f"\nQ{i}: {question}")
        lines.append(f"  Answer      : {str(answer)[:150]}...")
        lines.append(f"  Time        : {elapsed}s")
        lines.append(f"  Faithfulness: {faith}")
        lines.append(f"  Chunks used : {len(sources)}")

    lines += [
        "\n" + "-" * 55,
        f"  Avg time: {round(total_time / max(len(questions), 1), 2)}s",
        "=" * 55,
    ]
    return "\n".join(lines)
