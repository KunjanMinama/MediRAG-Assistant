from fastapi import APIRouter
from fastapi.responses import JSONResponse
from logger import logger
import os

router = APIRouter()

@router.post("/clear_index/")
async def clear_index():
    try:
        from pinecone import Pinecone
        import pickle
        
        pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        
        # Clear Pinecone
        index.delete(delete_all=True)
        logger.info("Pinecone index cleared ✅")
        
        # Clear BM25 model
        if os.path.exists("./bm25_model.pkl"):
            os.remove("./bm25_model.pkl")
            logger.info("BM25 model cleared ✅")

        # Clear uploaded docs
        import shutil
        if os.path.exists("./uploaded_docs"):
            shutil.rmtree("./uploaded_docs")
            os.makedirs("./uploaded_docs")
            logger.info("Uploaded docs cleared ✅")

        return {"message": "Index cleared successfully. Ready for new documents."}
    
    except Exception as e:
        logger.exception("Error clearing index")
        return JSONResponse(status_code=500, content={"error": str(e)})