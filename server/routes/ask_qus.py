from fastapi import APIRouter, Form
from typing import List, Optional
from module.llm import get_llm_chain
from module.quer_handler import query_chain
from module.bm25_encoder import load_bm25, encode_query
from module.reranker import Reranker
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from fastapi.responses import JSONResponse
from logger import logger
import os

router = APIRouter()
reranker = Reranker(api_key="exqG3LMd3pRmzjSeLHTiL0V6B7JIG4CJQRgWSztY")

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

        # Step 5.1: Rerank documents
        logger.debug("Reranking retrieved chunks...")
        retrieved_texts=[doc.page_content for doc in docs]

        reranked = reranker.rerank(
            query=question,
            documents= retrieved_texts,
            top_k=3
            )
        # Convert reranker top-3 chunks to final context
        final_context = "\n\n".join([item["text"] for item in reranked])

        logger.info(f"Reranker selected {len(reranked)}. Top chunks:")

    

        # Log scores for debugging
        for i, match in enumerate(res["matches"]):
            logger.debug(
                f"Chunk {i+1} | score: {match['score']:.4f} | "
                f"page: {match['metadata'].get('page', '')}"
            )

      
        logger.info("Generating answer using final reranked contexts...")

        prompt= f"""
        You are medical expert. Answer the question based on the following context:

        context:
        {final_context}
        question:
        {question}  
        """
         
        llm_chain=get_llm_chain(retriever=None) 
        answer=llm_chain.invoke({
            "context": final_context,
            "question": question
        })
        

        logger.info("Query successful")

        return {
            "response":         answer.content,
          
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