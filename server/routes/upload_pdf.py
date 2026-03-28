from fastapi import APIRouter, UploadFile, File
from typing import List
from module.load_vectorstores import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger

router=APIRouter()

@router.post("/upload_pdfs/")
async def upload_pdfs(files:List[UploadFile] = File(...)):
    try:
        logger.info("Recived uploaded files")
        load_vectorstore(files)
        logger.info("Document added to vectorstore")
        return {"message":"Files processed and vectorstore updated"}
    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500, content={"error":str(e)})