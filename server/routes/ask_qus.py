from fastapi import APIRouter, Form 
from typing import List, Optional
from module.load_vectorstores import load_vectorstore
from module.llm import get_llm_chain
from module.quer_handler import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from fastapi.responses import JSONResponse
from logger import logger
import os

router=APIRouter()

@router.post("/ask/")
async def ask_question(question:str=Form(...)):
    try:
        logger.info(f"user query:{question}")

        #embed model + pinecone setup
        pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        embedded_query=embed_model.embed_query(question)
        res=index.query(vector=embedded_query,top_k=8,include_metadata=True)
        
        docs=[
            Document(
                page_content=match["metadata"].get("text",""),
                metadata=match["metadata"]

            ) for match in res["matches"]
        ]
        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]]= Field(default_factory=list)
            metadata: Optional[dict]= Field(default_factory=dict)

            
            docs: List[Document] = Field(default_factory=list)  # ✅ declare it as a field

            def _get_relevant_documents(self, query: str) -> List[Document]:
                 return self.docs
            
        retriever = SimpleRetriever(docs=docs)
        chain=get_llm_chain(retriever)
        result=query_chain(chain,question)

        logger.info("Query is successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500,content={"error":str(e)})