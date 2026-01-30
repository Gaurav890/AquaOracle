"""Prompt templates for RAG system."""

from typing import List, Dict, Any


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context from source documents.

Key instructions:
1. Answer questions using ONLY information from the provided context
2. Always cite sources using [Document, Page X] format
3. If the information is not in the context, say "I don't have information about this in the provided documents."
4. Be precise and factual
5. Do not make up or infer information beyond what is explicitly stated
6. When multiple sources say similar things, cite all relevant sources
7. Keep answers concise but complete"""


def build_rag_prompt(
    question: str,
    context_chunks: List[Dict[str, Any]],
    max_context_length: int = 8000,
) -> str:
    """
    Build RAG prompt with question and context.

    Args:
        question: User's question
        context_chunks: List of retrieved chunks with metadata
        max_context_length: Maximum context length in characters

    Returns:
        Formatted prompt
    """
    # Build context section with source markers
    context_parts = []
    current_length = 0

    for i, chunk in enumerate(context_chunks, 1):
        doc_id = chunk.get("doc_id", "Unknown")
        page_numbers = chunk.get("page_numbers", [])
        text = chunk.get("text", "")

        # Format page numbers
        if page_numbers:
            if len(page_numbers) == 1:
                page_str = f"Page {page_numbers[0]}"
            else:
                page_str = f"Pages {page_numbers[0]}-{page_numbers[-1]}"
        else:
            page_str = "Page Unknown"

        # Create source marker
        source_marker = f"[{i}] Source: {doc_id}, {page_str}"

        # Check length
        chunk_text = f"\n{source_marker}\n{text}\n"
        if current_length + len(chunk_text) > max_context_length:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    context = "\n---\n".join(context_parts)

    # Build full prompt
    prompt = f"""Context from source documents:

{context}

---

Question: {question}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Cite your sources using [Source Number] format (e.g., [1], [2])
- If the answer is not in the context, say "I don't have information about this in the provided documents."
- Be specific and include relevant details
- List all sources used at the end under "Sources:"

Answer:"""

    return prompt


def build_citation_extraction_prompt(answer: str, sources: List[Dict[str, Any]]) -> str:
    """
    Build prompt to extract and validate citations from answer.

    Args:
        answer: Generated answer
        sources: Available source documents

    Returns:
        Prompt for citation extraction
    """
    sources_list = "\n".join([
        f"[{i+1}] {src.get('doc_id', 'Unknown')}, Pages {src.get('page_numbers', [])}"
        for i, src in enumerate(sources)
    ])

    prompt = f"""Given the following answer and available sources, extract all citations and verify they are correct.

Answer:
{answer}

Available Sources:
{sources_list}

Task:
1. List all citations used in the answer (e.g., [1], [2])
2. Verify each citation references a valid source
3. Extract the specific page numbers cited
4. Format the final "Sources:" section

Output format:
Sources:
[1] Document Name - Pages X-Y: "Section Title"
[2] Document Name - Page Z: "Section Title"
"""

    return prompt


def build_query_expansion_prompt(question: str) -> str:
    """
    Build prompt to expand user query with related terms.

    Args:
        question: Original question

    Returns:
        Prompt for query expansion
    """
    prompt = f"""Given the following question, generate 3-5 related search queries that would help find relevant information.

Original Question: {question}

Generate related queries that:
1. Use synonyms and related terms
2. Break down complex questions into sub-questions
3. Include alternative phrasings
4. Consider different aspects of the topic

Related Queries (one per line):"""

    return prompt


def build_answer_validation_prompt(question: str, answer: str, context: str) -> str:
    """
    Build prompt to validate answer against context.

    Args:
        question: Original question
        answer: Generated answer
        context: Source context

    Returns:
        Validation prompt
    """
    prompt = f"""Validate if the answer is supported by the context.

Question: {question}

Answer: {answer}

Context: {context}

Task:
1. Check if each claim in the answer is supported by the context
2. Identify any hallucinations or unsupported claims
3. Verify citations are accurate

Output:
- Validation: [PASS/FAIL]
- Issues: [List any problems]
- Confidence: [HIGH/MEDIUM/LOW]"""

    return prompt
