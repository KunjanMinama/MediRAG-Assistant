# MediRAG-Assistant
End-To-End Modular RAG Medical AI Assistant using LangChain, Pinecone, FastAPI.
🧠 RAG Medical AI Assistant

A lightweight Retrieval-Augmented Generation (RAG) Medical AI Assistant built using FastAPI, LangChain, FAISS, and HuggingFace.
Users can upload medical PDFs and ask questions, and the system retrieves relevant chunks + generates accurate AI responses.

🚀 Features
📄 PDF Uploading (FastAPI + UploadFile)
🧩 Text Splitting & Vector Embedding using HuggingFace embeddings
📁 FAISS Vector Store for fast similarity search
🔗 RAG Chain Setup (Retriever + LLM)
🤖 LLM Integration using HuggingFaceEndpoint
🩺 AI assistant designed for medical question answering


🗂 Project Structure
server/
│── module/
│   ├── llm.py               # LLM + prompt template
│   ├── pdf_handler.py       # Extracts text & handles uploads
│   ├── load_vectorstores.py # Loads/creates FAISS vectorstore
│   ├── query_handler.py     # RAG pipeline (query → retriever → LLM)
│── main.py                   # FastAPI entry point
│── exception_handler.py      # Custom exceptions (optional)
│── logger.py                 # Logging utility


⚙️ How It Works (Simple Flow)
1.User uploads a PDF
2.Text is split into chunks
3.Each chunk is embedded & stored in FAISS
4.User asks a medical question
5.Retriever finds the top relevant chunks
6.LLM generates a final answer using context
7.Response + source documents returned

