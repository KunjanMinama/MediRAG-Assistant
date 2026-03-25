import os
import shutill
from fastapi import UploadFile
import tempfile

UPLOAD_DIR = "./upload_docs"

def save_uploaded_files(files:list[UploadFile])-> list[str]:
    os.makedirs(UPLOAD_DIR,exist_ok=True)
    file_path=[]
    for file in files:
        temp_path=os.path.join(UPLOAD_DIR,file.filename)
        with open(temp_path,"wb") as f:
            shutill.copyfileobj(file.file,f)
        file_path.append(temp_path)

    return file_path