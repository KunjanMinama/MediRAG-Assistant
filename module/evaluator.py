
import asyncio
from concurrent.futures import ThreadPoolExecutor
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
from langchain_huggingface import HuggingFaceEndpointEmbeddings  # ✅ changed
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from dotenv import load_dotenv

load_dotenv()


def _get_ragas_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        n=1,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return LangchainLLMWrapper(llm)


def _get_ragas_embeddings():
    # ✅ API-based — no local sentence_transformers needed
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _safe_score(value, default: float = 0.0) -> float:
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


def _run_ragas_sync(dataset, metrics, ragas_llm, ragas_embeddings):
    """Run RAGAS in a fresh thread with its own event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False
        )
    finally:
        loop.close()


def run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str = None
) -> dict:
    try:
        logger.info("Starting RAGAS evaluation...")

        if not ground_truth:
            ground_truth = answer

        data = {
            "question":     [question],
            "answer":       [answer],
            "contexts":     [contexts],
            "ground_truth": [ground_truth]
        }
        dataset = Dataset.from_dict(data)

        ragas_llm        = _get_ragas_llm()
        ragas_embeddings = _get_ragas_embeddings()

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        # ✅ Run in separate thread to avoid nested event loop conflict
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_ragas_sync, dataset, metrics, ragas_llm, ragas_embeddings
            )
            result = future.result(timeout=120)  # 2 min timeout

        scores = {
            "faithfulness":      _safe_score(result["faithfulness"]),
            "answer_relevancy":  _safe_score(result["answer_relevancy"]),
            "context_recall":    _safe_score(result["context_recall"]),
            "context_precision": _safe_score(result["context_precision"]),
        }
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
    if score >= 0.8:
        return {"label": "Excellent", "color": "#27ae60"}
    elif score >= 0.6:
        return {"label": "Good",      "color": "#2e86c1"}
    elif score >= 0.4:
        return {"label": "Fair",      "color": "#e67e22"}
    else:
        return {"label": "Poor",      "color": "#c0392b"}
