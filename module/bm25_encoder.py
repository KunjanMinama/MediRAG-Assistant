import json
import pickle
from pathlib import Path
from pinecone_text.sparse import BM25Encoder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import logger

# Path to save fitted BM25 model
BM25_PATH = "./bm25_model.pkl"


# ─────────────────────────────────────────────────────────────────
# FIT AND SAVE BM25
# Called during PDF upload — learns vocabulary from your documents
# Must be fit before encoding queries
# ─────────────────────────────────────────────────────────────────
def fit_and_save_bm25(texts: list[str]) -> BM25Encoder:
    """
    Fit BM25 on document chunks and save to disk.
    
    Why save to disk?
    Because BM25 needs to know the vocabulary of ALL documents
    to calculate IDF scores. We save it so queries use same vocab.
    """
    try:
        logger.info(f"Fitting BM25 on {len(texts)} text chunks...")

        bm25 = BM25Encoder.default()
        bm25.fit(texts)  # learns vocabulary + IDF scores

        # Save to disk so we can load it later for queries
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

        logger.info(f"BM25 model saved to {BM25_PATH}")
        return bm25

    except Exception as e:
        logger.exception("Failed to fit BM25")
        raise


# ─────────────────────────────────────────────────────────────────
# LOAD SAVED BM25
# Called during query — loads the fitted model from disk
# ─────────────────────────────────────────────────────────────────
def load_bm25() -> BM25Encoder:
    """
    Load previously fitted BM25 model from disk.
    Raises error if model not found (upload PDFs first).
    """
    if not Path(BM25_PATH).exists():
        raise FileNotFoundError(
            "BM25 model not found! Please upload PDFs first to fit the model."
        )

    try:
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        logger.debug("BM25 model loaded successfully")
        return bm25

    except Exception as e:
        logger.exception("Failed to load BM25 model")
        raise


# ─────────────────────────────────────────────────────────────────
# ENCODE DOCUMENTS (for upsert)
# Converts text chunks to sparse vectors for Pinecone storage
# ─────────────────────────────────────────────────────────────────
def encode_documents(bm25: BM25Encoder, texts: list[str]) -> list[dict]:
    """
    Convert text chunks to sparse vectors.
    Returns list of sparse dicts like: {"indices": [...], "values": [...]}
    """
    try:
        sparse_vectors = bm25.encode_documents(texts)
        logger.debug(f"Encoded {len(texts)} documents to sparse vectors")
        return sparse_vectors
    except Exception as e:
        logger.exception("Failed to encode documents")
        raise


# ─────────────────────────────────────────────────────────────────
# ENCODE QUERY (for search)
# Converts query to sparse vector for Pinecone hybrid search
# ─────────────────────────────────────────────────────────────────
def encode_query(bm25: BM25Encoder, query: str) -> dict:
    """
    Convert query to sparse vector.
    Returns sparse dict like: {"indices": [...], "values": [...]}
    """
    try:
        sparse_vector = bm25.encode_queries(query)
        logger.debug(f"Query encoded to sparse vector")
        return sparse_vector
    except Exception as e:
        logger.exception("Failed to encode query")
        raise