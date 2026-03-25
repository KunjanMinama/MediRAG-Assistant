import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV ="us-east-1"
PINECONE_INDEX_NAME="medical_index"

os.environ["HUGGINGFACE_API_KEY"]=HUGGINGFACE_API_KEY

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)

# Initialize pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec

    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)

## Load,split,embeed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embed_model=HuggingFaceEmbeddings(model="BAAI/bge-small-en")
    file_paths=[]
    
    # 1. upload
    for file in uploaded_files:
        save_path=Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    #2. split
    for file_path in file_paths:
        loader=PyPDFLoader(file_path)
        document=loader.laod()

        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        chunks=splitter.split_documents(document)
        
        texts=[ chunks.page_content for chunk in chunks]
        metadata = [chunks.metadata for chunk in chunks]
        ids=[f"{Path(file_path).stem}-{i}" for i in range(chunks)]

        #3 embedding
        print(f"Embedding chunks")
        embedding=embed_model.embed_documents(texts)

        # 4. Upsert
        print("Userting embeddings..")
        with tqdm(total=len(embedding),desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids,embedding,metadata))

        print(f"Upload complete for {file_path}")
                         