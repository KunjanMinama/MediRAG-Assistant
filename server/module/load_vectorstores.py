import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from module.bm25_encoder import encode_documents, fit_and_save_bm25
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from logger import logger

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV ="us-east-1"
PINECONE_INDEX_NAME="meeddiassist"

os.environ["HUGGINGFACE_API_TOKEN"]=HUGGINGFACE_API_KEY

UPLOAD_DIR="./uploaded_docs"  ##ALl the uploaded PDFs will store here
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
    embed_model=HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
                                      )
    file_paths=[]
    logger.info("Pinecone index ready, starting to process uploaded files")

    
    # 1. upload
    for file in uploaded_files:
        save_path=Path(UPLOAD_DIR)/file.filename
        with open(save_path,"wb") as f: 
            f.write(file.file.read())
        file_paths.append(str(save_path))
        logger.info(f"Saved file: {save_path}")

    all_chunks=[]
    all_texts=[]
    all_ids=[]
    all_metadata=[]

    #2. split
    for file_path in file_paths:
        logger.info(f"Processing: {file_path}")

        reader = PdfReader(file_path)
        document = [
             Document(
              page_content=page.extract_text() or "",
              metadata={"source": file_path, "page": i}
    )
             for i, page in enumerate(reader.pages)
]
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=splitter.split_documents(document)
        
         # Collect texts, ids, metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{Path(file_path).stem}-{i}"
            all_chunks.append(chunk)
            all_texts.append(chunk.page_content)
            all_ids.append(chunk_id)
            all_metadata.append({
                **chunk.metadata,
                "text": chunk.page_content  # store text for retrieval
            })
 
    logger.info(f"Total chunks to process: {len(all_chunks)}")

    # Generate dense embeddings ────────────────────────────────────
    logger.info("Generating dense embeddings...")
    dense_embeddings = embed_model.embed_documents(all_texts)
    logger.info(f"Dense embeddings generated: {len(dense_embeddings)}")

    #  Fit BM25 and generate sparse vectors ─────────────────────────
    # NEW: fit BM25 on ALL texts so it learns the vocabulary
    logger.info("Fitting BM25 and generating sparse vectors...")
    bm25           = fit_and_save_bm25(all_texts)  # learns vocab + IDF scores
    sparse_vectors = encode_documents(bm25, all_texts)  # learns vocab + IDF, returns sparse vectors
    logger.info(f"Sparse vectors generated: {len(sparse_vectors)}")
 
    # Upsert to Pinecone with BOTH dense + sparse ──────────────────
    logger.info("Upserting to Pinecone with hybrid vectors...")
 
    vectors = [
        {
            "id":           all_ids[i],
            "values":       dense_embeddings[i],      # dense vector
            "sparse_values": sparse_vectors[i],        # sparse vector ← NEW
            "metadata":     all_metadata[i]
        }
        for i in range(len(all_chunks))
    ]
 
    with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
        for v in vectors:
            index.upsert(vectors=[v])
            progress.update(1)
 
    logger.info(f"Upload complete! {len(vectors)} vectors upserted.")