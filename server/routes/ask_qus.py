from fastapi import APIRouter, Form
from typing import List, Optional
from module.llm import get_llm_chain
from module.quer_handler import query_chain
from module.bm25_encoder import load_bm25, encode_query
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from fastapi.responses import JSONResponse
from logger import logger
import os

router = APIRouter()

HYBRID_ALPHA = 0.5  # 0.5 = equal dense + sparse balance


class SimpleRetriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        # Step 1: Setup
        pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # Step 2: Dense embedding
        logger.debug("Generating dense embedding...")
        dense_vector = embed_model.embed_query(question)

        # Step 3: Sparse BM25 vector
        logger.debug("Generating sparse BM25 vector...")
        bm25          = load_bm25()
        sparse_vector = encode_query(bm25, question)

        # Step 4: Hybrid search in Pinecone
        logger.debug(f"Hybrid search alpha={HYBRID_ALPHA}...")
        res = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=8,
            include_metadata=True,
            alpha=HYBRID_ALPHA
        )

        logger.info(f"Retrieved {len(res['matches'])} chunks")

        # Step 5: Build documents
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        # Log scores for debugging
        for i, match in enumerate(res["matches"]):
            logger.debug(
                f"Chunk {i+1} | score: {match['score']:.4f} | "
                f"page: {match['metadata'].get('page', '')}"
            )

        # Step 6: Build chain and run
        retriever = SimpleRetriever(docs=docs)
        chain     = get_llm_chain(retriever)
        result    = query_chain(chain, question)

        logger.info("Query successful")

        return {
            "response":         result["response"],
            "original_query":   result.get("original_query", question),
            "rewritten_query":  result.get("rewritten_query", question),
            "chunks_retrieved": len(docs),
            "hybrid_alpha":     HYBRID_ALPHA
        }

    except FileNotFoundError as e:
        logger.warning(f"BM25 not found: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload PDF documents first."}
        )
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})