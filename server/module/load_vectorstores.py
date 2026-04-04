# ✅ CORRECT load_vectorstores.py
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from logger import logger
import pickle

load_dotenv()

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_ENV        = "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "meddiassist")
UPLOAD_DIR          = "./uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ NO module level Pinecone connection — everything inside function!

def load_vectorstore(uploaded_files):
    # Init Pinecone INSIDE function only
    pc   = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    existing_indexes = [i["name"] for i in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=spec
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    index       = pc.Index(PINECONE_INDEX_NAME)
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_paths  = []

    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
        logger.info(f"Saved: {save_path}")

    all_texts    = []
    all_ids      = []
    all_metadata = []

    for file_path in file_paths:
        reader   = PdfReader(file_path)
        document = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": file_path, "page": i}
            )
            for i, page in enumerate(reader.pages)
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = splitter.split_documents(document)
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk.page_content)
            all_ids.append(f"{Path(file_path).stem}-{i}")
            all_metadata.append({
                **chunk.metadata,
                "text": chunk.page_content
            })

    logger.info(f"Total chunks: {len(all_texts)}")

    dense = embed_model.embed_documents(all_texts)

    bm25 = BM25Encoder.default()
    bm25.fit(all_texts)
    with open("./bm25_model.pkl", "wb") as f:
        pickle.dump(bm25, f)
    sparse = bm25.encode_documents(all_texts)

    vectors = [
        {
            "id":            all_ids[i],
            "values":        dense[i],
            "sparse_values": sparse[i],
            "metadata":      all_metadata[i]
        }
        for i in range(len(all_texts))
    ]

    with tqdm(total=len(vectors), desc="Upserting") as p:
        for v in vectors:
            index.upsert(vectors=[v])
            p.update(1)

    logger.info(f"Done! {len(vectors)} vectors upserted.")