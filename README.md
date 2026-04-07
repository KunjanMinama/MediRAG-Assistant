# 🏥 MediRAG — AI Medical Document Assistant

<div align="center">

![MediRAG Banner](https://img.shields.io/badge/MediRAG-AI%20Medical%20Assistant-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)

**Upload medical reports. Ask questions. Get accurate, grounded answers — powered by Advanced RAG + LLaMA 3.**

[🚀 Live Demo](https://kunjanminama-medirag-assistant-clientapp-ttrgwz.streamlit.app) • [🔧 Backend API](https://kunjan174-medirag-backend.hf.space) • [📖 API Docs](https://kunjan174-medirag-backend.hf.space/docs)

</div>

---

## 📌 What is MediRAG?

MediRAG is a production-deployed **Retrieval-Augmented Generation (RAG)** system built for medical document Q&A. Upload patient PDFs, ask natural language questions, and get precise answers grounded strictly in the uploaded documents — no hallucination.

> ⚠️ **Disclaimer**: MediRAG is an AI assistant for document understanding only. It does not provide medical advice or diagnosis.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-PDF Upload** | Upload multiple patient reports simultaneously |
| 🔍 **Hybrid Search** | Dense (semantic) + Sparse (BM25) retrieval for maximum accuracy |
| 🎯 **Cohere Reranking** | Re-ranks retrieved chunks for highest relevance |
| 🧠 **Multi-Doc Reasoning** | MAP-Reduce chain for comparing across multiple documents |
| 🤖 **Simple RAG** | Direct retrieval chain for single-document queries |
| 📊 **RAGAS Evaluation** | Real-time RAG quality metrics (Faithfulness, Relevancy, Precision, Recall) |
| 🗑️ **Clear Index** | One-click reset to switch between patient documents |
| 💬 **Chat History** | Download conversation history as TXT, JSON, or Markdown |
| 🚫 **Anti-Hallucination** | Strict prompt engineering — never uses general medical knowledge |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│         (Chat UI + RAGAS Dashboard + Uploader)          │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP REST API
┌─────────────────────▼───────────────────────────────────┐
│              FastAPI Backend (HuggingFace Spaces)        │
│                                                          │
│  /upload_pdfs/  →  PDF Processing + Vectorization        │
│  /ask/          →  Hybrid Search + Rerank + LLM          │
│  /evaluate/     →  RAGAS Evaluation Pipeline             │
│  /clear_index/  →  Reset Pinecone Index                  │
└──────┬──────────────────────────┬────────────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  Pinecone   │          │  HuggingFace    │
│ Vector DB   │          │  Inference API  │
│ (Hybrid     │          │  (LLaMA 3.1 8B) │
│  Search)    │          └─────────────────┘
└─────────────┘
```

---

## 🔬 RAG Pipeline

```
PDF Upload
    │
    ▼
Text Extraction (PyPDF)
    │
    ▼
Document Chunking (RecursiveCharacterTextSplitter)
    │
    ├──► Dense Embeddings  (HuggingFace all-MiniLM-L6-v2)
    │
    └──► Sparse Embeddings (BM25Encoder)
              │
              ▼
         Pinecone Hybrid Upsert
              │
         [At Query Time]
              │
    User Question
         │
         ├──► Dense Query Vector
         │
         └──► BM25 Sparse Vector
                   │
                   ▼
            Pinecone Hybrid Search (α=0.5)
                   │
                   ▼
            Cohere Reranker (top 5)
                   │
                   ├──► Single Source? → Simple RAG Chain
                   │
                   └──► Multi Source?  → MAP-Reduce Chain
                                │
                                ▼
                          LLaMA 3.1 8B (via HuggingFace)
                                │
                                ▼
                          Final Answer
```

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **FastAPI** | REST API framework |
| **Pinecone** | Vector database (hybrid search) |
| **LangChain** | RAG orchestration |
| **LLaMA 3.1 8B** | LLM via HuggingFace Inference API |
| **Cohere Rerank** | Result reranking |
| **BM25Encoder** | Sparse vector encoding |
| **RAGAS** | RAG evaluation metrics |
| **HuggingFace Spaces** | Backend deployment |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Frontend framework |
| **Streamlit Cloud** | Frontend deployment |

---

## 📊 RAGAS Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Were the retrieved chunks relevant? |
| **Context Recall** | Did retrieval find all relevant information? |

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.11+
Pinecone account (free tier)
HuggingFace account (free tier)
Cohere account (free tier)
Groq account (free tier)
```

### Environment Variables
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
HUGGINGFACE_API_TOKEN=your_hf_token
COHERE_API_KEY=your_cohere_key
GROQ_API_KEY=your_groq_key
```

### Backend Setup
```bash
git clone https://github.com/KunjanMinama/MediRAG-Assistant.git
cd MediRAG-Assistant/server

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd MediRAG-Assistant/client

pip install streamlit requests

streamlit run app.py
```

---

## 📁 Project Structure

```
MediRAG-Assistant/
├── server/                          # FastAPI Backend
│   ├── main.py                      # App entry point
│   ├── requirements.txt
│   ├── Dockerfile                   # HuggingFace deployment
│   ├── routes/
│   │   ├── upload_pdf.py            # PDF upload endpoint
│   │   ├── ask_qus.py               # Q&A endpoint
│   │   ├── evaluate.py              # RAGAS evaluation endpoint
│   │   └── clear_index.py           # Index reset endpoint
│   ├── module/
│   │   ├── load_vectorstores.py     # PDF processing + Pinecone upsert
│   │   ├── llm.py                   # LLM chain setup
│   │   ├── multidoc_chain.py        # MAP-Reduce reasoning
│   │   ├── reranker.py              # Cohere reranking
│   │   ├── bm25_encoder.py          # Sparse encoding
│   │   ├── evaluator.py             # RAGAS pipeline
│   │   └── quer_handler.py          # Query processing
│   └── middleware/
│       └── exception_handlers.py
│
└── client/                          # Streamlit Frontend
    ├── app.py                       # Main app
    └── components/
        ├── chatUI.py                # Chat interface
        ├── upload.py                # File uploader
        ├── ragas_dashboard.py       # Evaluation dashboard
        └── history_downloader.py   # Chat history export
```

---

## 🌐 Deployment

| Service | Platform | URL |
|---------|----------|-----|
| Backend API | HuggingFace Spaces (Free) | `kunjan174-medirag-backend.hf.space` |
| Frontend | Streamlit Cloud (Free) | `kunjanminama-medirag-assistant.streamlit.app` |
| Vector DB | Pinecone (Free Tier) | Serverless, AWS us-east-1 |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use this project for learning and portfolio purposes.

---

## 👨‍💻 Author

**Kunjan Minama**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/KunjanMinama)

---

<div align="center">
⭐ Star this repo if you found it helpful!
</div>
