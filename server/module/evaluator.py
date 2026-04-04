import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────
# RAGAS EVALUATION
#
# Judge LLM  : Groq  (free, fast, reliable — no routing issues)
# Embeddings : HuggingFace sentence-transformers (free, local)
#
# RAGAS needs 4 inputs per question:
# 1. question        - the user's question
# 2. answer          - the LLM's answer
# 3. contexts        - list of retrieved chunks used
# 4. ground_truth    - the ideal answer (for recall metric)
#
# Metrics scored:
# - Faithfulness      → is answer grounded in contexts?
# - Answer Relevancy  → does answer address the question?
# - Context Recall    → did we find all relevant info?
# - Context Precision → were retrieved chunks relevant?
# ─────────────────────────────────────────────────────────────────


def _get_ragas_llm():
    """
    Groq-hosted LLaMA3 as the RAGAS judge LLM.
    - Free tier available at https://console.groq.com
    - Fast inference, no routing issues
    - Requires GROQ_API_KEY in your .env
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",      # best free Groq model for evaluation
        temperature=0.0,              # deterministic scoring
        n=1,
        api_key=os.getenv("GROQ_API_KEY"),
       
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    """
    Local HuggingFace embeddings for answer relevancy metric.
    Runs on CPU, no API key needed.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _safe_score(value, default: float = 0.0) -> float:
    """
    Safely extract a finite float from whatever RAGAS returns.
    """
    try:
        if isinstance(value, (list, tuple)):
            numeric = []
            for v in value:
                try:
                    f = float(v)
                    if not math.isnan(f) and not math.isinf(f):
                        numeric.append(f)
                except (TypeError, ValueError):
                    pass
            value = (sum(numeric) / len(numeric)) if numeric else default

        if value is None:
            return default

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return round(value, 3)

    except (TypeError, ValueError):
        return default


def run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = None
) -> dict:
    """ 
    Run RAGAS evaluation on a single QA pair.

    Args:
        question:     The user's question
        answer:       The LLM's generated answer
        contexts:     List of retrieved text chunks used as context
        ground_truth: The ideal answer (optional — needed for context_recall)

    Returns:
        dict with scores for each metric (0.0 to 1.0), all JSON-safe floats
    """
    try:
        logger.info("Starting RAGAS evaluation...")

        if not ground_truth:
            ground_truth = answer
            logger.warning("No ground truth provided — using answer as ground truth")

        data = {
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)
        logger.debug(f"RAGAS dataset created with {len(contexts)} contexts")

        ragas_llm        = _get_ragas_llm()
        ragas_embeddings = _get_ragas_embeddings()

        logger.info("Running RAGAS metrics evaluation via Groq (llama3-70b-8192)...")
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False
        )

        scores = {
            "faithfulness":      _safe_score(result["faithfulness"]),
            "answer_relevancy":  _safe_score(result["answer_relevancy"]),
            "context_recall":    _safe_score(result["context_recall"]),
            "context_precision": _safe_score(result["context_precision"]),
        }

        failed_metrics = [k for k, v in scores.items() if v == 0.0]
        if failed_metrics:
            logger.warning(
                "The following metrics scored 0.0 (likely due to eval job "
                "failures): %s", failed_metrics
            )

        scores["overall"] = round(sum(scores.values()) / len(scores), 3)

        logger.info(f"RAGAS evaluation complete: {scores}")
        return scores

    except Exception as e:
        logger.exception("RAGAS evaluation failed")
        return {
            "faithfulness":      0.0,
            "answer_relevancy":  0.0,
            "context_recall":    0.0,
            "context_precision": 0.0,
            "overall":           0.0,
            "error":             str(e)
        }


def get_score_label(score: float) -> dict:
    """
    Returns label and color for a score value.
    Used by frontend to show colored badges.
    """
    if score >= 0.8:
        return {"label": "Excellent", "color": "#27ae60"}
    elif score >= 0.6:
        return {"label": "Good",      "color": "#2e86c1"}
    elif score >= 0.4:
        return {"label": "Fair",      "color": "#e67e22"}
    else:
        return {"label": "Poor",      "color": "#c0392b"}