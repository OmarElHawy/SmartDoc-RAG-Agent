import re

HARD_BLOCK_PATTERNS = [
    r"ignore\s+(previous|all|your|prior)\s+(instructions?|prompts?|rules?|context)",
    r"(you are now|pretend you are|act as if you are|forget you are)",
    r"(jailbreak|bypass|override|disable)\s+(your|the|all)?\s*(safety|filter|guardrail|restriction|rule)",
    r"(generate|write|create|produce)\s+.{0,30}(malware|virus|exploit|ransomware|keylogger|trojan)",
    r"(hack|exploit|attack|crack|break into)\s+\w+",
    r"how\s+(do\s+i|to)\s+(make|build|create|synthesize)\s+.{0,30}(bomb|weapon|explosive|poison|drug)",
    r"(give me|show me|provide)\s+.{0,30}(illegal|stolen|confidential|classified)\s+(data|info|document|access)",
]

CHIT_CHAT_PATTERNS = [
    r"^(hi+|hello+|hey+|sup|yo|howdy|hiya|what'?s\s+up)[!?.]*$",
    r"^(good\s+(morning|afternoon|evening|night))[!?.]*$",
    r"(tell\s+me\s+a\s+joke|make\s+me\s+laugh|say\s+something\s+funny)",
    r"(sing\s+(me\s+)?a\s+song|write\s+(me\s+)?a\s+poem|write\s+(me\s+)?a\s+haiku)",
    r"(what'?s?\s+(the\s+)?(current\s+)?(weather|temperature|forecast)\s+(in|today|now|like))",
    r"(what\s+(time|day|date)\s+is\s+it)",
    r"(who\s+(are\s+you|made\s+you|created\s+you|built\s+you|is\s+your\s+(creator|developer|maker)))",
    r"(what'?s?\s+your\s+(name|favorite|opinion\s+on|hobby|age))",
    r"(do\s+you\s+(like|love|hate|enjoy|have\s+feelings|feel))",
    r"(are\s+you\s+(human|a\s+robot|sentient|alive|conscious|real))",
    r"(recommend\s+(me\s+)?(a\s+)?(movie|show|series|game|book|song|restaurant)\s+to)",
    r"(how\s+(do\s+i\s+)?(cook|bake|make|prepare)\s+\w+)",   # cooking questions
    r"(what'?s?\s+the\s+(score|result)\s+of\s+.{0,30}(game|match|fight))",
    r"(who\s+won\s+the\s+(super\s+bowl|world\s+cup|championship|election))",
    r"(translate\s+.{0,30}\s+to\s+(arabic|french|spanish|german|chinese|japanese))",
]


class SemanticGuardrail:
    """
    A lightweight, document-agnostic guardrail that uses regex pattern matching
    to block two categories of bad queries:

      1. HARD BLOCKS  — harmful, adversarial, or prompt-injection content.
      2. CHIT-CHAT    — clearly conversational with no relevance to any document.

    Everything else is allowed. Factuality and relevance to the *specific*
    document is already handled by the RAG system prompt grounding, which
    instructs the LLM to say "The document does not contain information about
    [topic]" when the answer isn't in the retrieved chunks.
    """

    def check(self, query: str) -> dict:
        """
        Evaluate a query against the guardrail rules.

        Returns:
            dict with keys:
                - is_allowed (bool)
                - score      (float: 0.0 = blocked, 1.0 = allowed)
                - reason     (str: human-readable explanation)
        """
        if not query or not query.strip():
            return {
                "is_allowed": True,
                "score": 1.0,
                "reason": "Empty query passed through.",
            }

        q = query.lower().strip()

        for pattern in HARD_BLOCK_PATTERNS:
            if re.search(pattern, q):
                print(f"[Guardrails] HARD BLOCK matched pattern '{pattern}' on query: '{query}'")
                return {
                    "is_allowed": False,
                    "score": 0.0,
                    "reason": "Query contains harmful, adversarial, or prompt-injection content.",
                }

        for pattern in CHIT_CHAT_PATTERNS:
            if re.search(pattern, q):
                print(f"[Guardrails] CHIT-CHAT matched pattern '{pattern}' on query: '{query}'")
                return {
                    "is_allowed": False,
                    "score": 0.0,
                    "reason": (
                        "Your question appears to be general chit-chat unrelated to the document. "
                        "Please ask something about the uploaded document."
                    ),
                }

        print(f"[Guardrails] ALLOWED query: '{query}'")
        return {
            "is_allowed": True,
            "score": 1.0,
            "reason": "Query allowed — will be answered from document context.",
        }


_guardrail_instance = None


def get_guardrail() -> SemanticGuardrail:
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = SemanticGuardrail()
        print("[Guardrails] Pattern-based guardrail initialized (no training required).")
    return _guardrail_instance