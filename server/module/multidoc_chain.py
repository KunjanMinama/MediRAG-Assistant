import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

#-------------------------------------------------------------------------
#MUlti-Document REASONING using Map-Reduce Pattern

## How its Works:
## MAP STEP : each document chunks is summarized individually by
#             the LLM to exttract relevant information
#
#REDUCE STEP: ALL individual summaries are combined together and 
#             passed to LLM for final comprehensive answer
#-------------------------------------------------------------------------


def get_llm():
    # ✅ Moved here — only loads torch when first request comes in
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="chat-completion",
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )
    return ChatHuggingFace(llm=llm)

#--------------------------------------------------------------------------
#MAP STEP: Summarize each document chunk individually to extract relevant info
#--------------------------------------------------------------------------

def map_step(docs: list[Document], question: str, llm) -> list[str]:
    """
    Procces each dc=ocumrnt chunk individually.
    Extracr relevatn info for the given question.
    Returns Llist of summaries - One per documents """

    map_prompt=PromptTemplate(
        input_variables=["context","question"],
        template="""You are a medical document analyst.
Extract ALL medical information from this document chunk
that could be useful for answering the question.
Include patient name, age, diagnosis, lab results,
medications, and any other relevant medical data.
Even if not directly related, extract key medical facts.
        Document Chunk:
        {context}

        Question:
        {question}

        """
    )
    summaries=[]

    for i, doc in enumerate(docs):
        try:
            logger.debug(f"MAP step — processing chunk {i+1}/{len(docs)}")
 
            # Format prompt with this specific chunk
            prompt_text = map_prompt.format(
                context=doc.page_content,
                question=question
            )
 
            # Get summary from LLM
            result = llm.invoke(prompt_text)
            summary = result.content.strip()
 
            summaries.append(summary)
            logger.debug(f"Chunk {i+1} summary: {summary[:80]}...")
 
        except Exception as e:
            logger.warning(f"MAP step failed for chunk {i+1}: {e}")
            summaries.append(f"Could not process chunk {i+1}")
 
    logger.info(f"MAP step complete — {len(summaries)} summaries generated")
    return summaries
 
 
# ─────────────────────────────────────────────────────────────────
# REDUCE STEP
# Combine all summaries into one final comprehensive answer
# This is where multi-document comparison happens
# ─────────────────────────────────────────────────────────────────
def _reduce_step(summaries: list[str], question: str, llm) -> str:
    """
    Combine all document summaries into a final answer.
    This enables comparison and reasoning across all documents.
    """
 
    # Join all summaries with document labels
    combined = "\n\n".join([
        f"Document {i+1} Information:\n{summary}"
        for i, summary in enumerate(summaries)
    ])
 
    reduce_prompt = PromptTemplate(
        input_variables=["combined_summaries", "question"],
        template="""You are MediBOT, an AI medical assistant.
Using the information extracted from multiple medical documents below,
provide a comprehensive and accurate answer to the question.
 
If comparing across documents, highlight similarities and differences.
If information is only in one document, mention which one.
Do not make up facts. Do not give medical advice or diagnoses.
 
Extracted Information from Documents:
{combined_summaries}
 
Question: {question}
 
Comprehensive Answer:"""
    )
 
    try:
        logger.debug("REDUCE step — combining all summaries...")
 
        prompt_text = reduce_prompt.format(
            combined_summaries=combined,
            question=question
        )
 
        result = llm.invoke(prompt_text)
        final_answer = result.content.strip()
 
        logger.info("REDUCE step complete — final answer generated")
        return final_answer
 
    except Exception as e:
        logger.exception("REDUCE step failed")
        raise
 
 
# ─────────────────────────────────────────────────────────────────
# MAIN FUNCTION: run_multidoc_chain
# Called from ask_qus.py
# Decides whether to use Map-Reduce or simple single-doc reasoning
# ─────────────────────────────────────────────────────────────────
def run_multidoc_chain(docs: list[Document], question: str) -> dict:
    """
    Run multi-document reasoning using Map-Reduce pattern.
 
    Args:
        docs:     List of retrieved Document objects (from all sources)
        question: User's question
 
    Returns:
        dict with response, summaries, and source info
    """
 
    logger.info(f"Starting multi-doc reasoning on {len(docs)} chunks")
 
    # Get unique source documents
    sources = list(set([
        doc.metadata.get("source", "Unknown")
        for doc in docs
    ]))
    logger.info(f"Sources involved: {sources}")
 
    llm = get_llm()
 
    # ── MAP STEP: summarize each chunk ────────────────────────────
    summaries = map_step(docs, question, llm)
 
    # ── REDUCE STEP: combine all summaries ────────────────────────
    final_answer = _reduce_step(summaries, question, llm)
 
    return {
        "response":         final_answer,
        "summaries":        summaries,          # intermediate map results
        "sources_involved": sources,            # which docs were used
        "chunks_processed": len(docs),
        "reasoning_type":   "map_reduce"
    }
 
 
# ─────────────────────────────────────────────────────────────────
# HELPER: Check if multi-doc reasoning is needed
# Use Map-Reduce when multiple source documents are involved
# Use simple RAG when only one source document
# ─────────────────────────────────────────────────────────────────
def needs_multidoc_reasoning(docs: list[Document]) -> bool:
    """
    Returns True if chunks come from multiple different source files.
    In that case we use Map-Reduce instead of simple RAG.
    """
    sources = set([
        doc.metadata.get("source", "")
        for doc in docs
    ])
    is_multi = len(sources) > 1
    logger.debug(f"Sources found: {sources} | Multi-doc needed: {is_multi}")
    return is_multi