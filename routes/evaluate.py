from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from module.evaluator import run_ragas_evaluation
from logger import logger
import json
import math

router = APIRouter()


def _make_json_safe(obj):
    """
    Recursively walk a dict/list and replace any non-JSON-compliant
    float values (nan, inf, -inf) with None.

    Python's json.dumps raises ValueError on these; this ensures the
    response always serializes cleanly regardless of what RAGAS returns.
    """
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None          # JSON null — safe to serialize
        return obj 
    return obj


@router.post("/evaluate/")
async def evaluate_response(
    question:     str = Form(...),
    answer:       str = Form(...),
    contexts:     str = Form(...),   # JSON string of list
    ground_truth: str = Form(None)
):
    """
    Evaluate a RAG response using RAGAS metrics.

    Accepts:
    - question:     The user's question
    - answer:       The LLM's generated answer
    - contexts:     JSON string of list of retrieved text chunks
    - ground_truth: The ideal answer (optional)

    Returns:
    RAGAS scores for all 4 metrics + overall score
    """
    try:
        logger.info(f"Evaluation request for question: '{question[:60]}...'")

        # ── Parse contexts ────────────────────────────────────────
        contexts_list = []
        try:
            parsed = json.loads(contexts)
            contexts_list = parsed if isinstance(parsed, list) else [contexts]
        except json.JSONDecodeError:
            contexts_list = [contexts]

        logger.debug(f"evaluating with {len(contexts_list)} context chunks")

        # ── Run RAGAS evaluation ──────────────────────────────────
        scores = run_ragas_evaluation(
            question=question,
            answer=answer,
            contexts=contexts_list,
            ground_truth=ground_truth if ground_truth else ""
        )

        logger.info(f"Evaluation completed with scores: {scores.get('overall', 0)}")

        # ── Build response (sanitize before JSON serialization) ───
        payload = _make_json_safe({
            "question":         question,
            "scores":           scores,
            "contexts_used":    len(contexts_list),
            "has_ground_truth": bool(ground_truth),
        })

        return JSONResponse(content=payload)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "An error occurred during evaluation. Please try again."}
        )