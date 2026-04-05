from fastapi import APIRouter, Form
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from fastapi.responses import JSONResponse
from logger import logger
import os

router = APIRouter()

HYBRID_ALPHA = 0.5
TOP_K = 10
TOP_N = 5

class SimpleRetriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        # ✅ All heavy imports moved here
        from module.llm import get_llm_chain
        from module.bm25_encoder import load_bm25, encode_query
        from module.reranker import Reranker
        from module.multidoc_chain import run_multidoc_chain, needs_multidoc_reasoning
        from module.quer_handler import query_chain
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        from pinecone import Pinecone

        logger.info(f"User query: {question}")

        pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model = HuggingFaceEndpointEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                    api_key=os.environ["HUGGINGFACE_API_KEY"])

        logger.debug("Generating dense embedding...")
        dense_vector = embed_model.embed_query(question)

        logger.debug("Generating sparse BM25 vector...")
        bm25          = load_bm25()
        sparse_vector = encode_query(bm25, question)

        logger.debug(f"Hybrid search alpha={HYBRID_ALPHA}...")
        res = index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=TOP_K,
            include_metadata=True,
            alpha=HYBRID_ALPHA
        )

        logger.info(f"Retrieved {len(res['matches'])} chunks")

        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            )
            for match in res["matches"]
        ]

        for i, match in enumerate(res["matches"]):
            logger.debug(
                f"Chunk {i+1} | score: {match['score']:.4f} | "
                f"page: {match['metadata'].get('page', '')}"
            )

        logger.debug("Reranking retrieved chunks...")
        reranker   = Reranker(api_key=os.environ["COHERE_API_KEY"])
        docs_texts = [doc.page_content for doc in docs]
        reranked   = reranker.rerank(question, docs_texts, top_k=TOP_N)

        reranked_docs = []
        for r in reranked:
            original_doc = next(
                (doc for doc in docs if doc.page_content == r["text"]), None
            )
            reranked_docs.append(
                Document(
                    page_content=r["text"],
                    metadata={
                        **(original_doc.metadata if original_doc else {}),
                        "rerank_score": r["score"]
                    }
                )
            )

        logger.info(f"Reranker selected {len(reranked_docs)} chunks.")

        if needs_multidoc_reasoning(reranked_docs):
            logger.info("Multiple sources detected - using MAP-Reduce chain")
            result = run_multidoc_chain(reranked_docs, question)
            return {"response": result["response"]}
        else:
            logger.info("Single source detected — using simple RAG chain")
            retriever = SimpleRetriever(docs=reranked_docs)
            chain     = get_llm_chain(retriever)
            result    = query_chain(chain, question)
            return {"response": result["response"]}

    except FileNotFoundError as e:
        logger.warning(f"BM25 not found: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload PDF documents first."}
        )
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})