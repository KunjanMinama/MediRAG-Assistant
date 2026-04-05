import nltk
import os

# ✅ Download to a path that persists at runtime
nltk.download('punkt_tab', download_dir='/opt/render/nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir='/opt/render/nltk_data')
nltk.download('stopwords', download_dir='/opt/render/nltk_data')  # often needed too

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# They will be imported only when routes are called
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

log.info("=== STARTING IMPORTS ===")

from fastapi import FastAPI
log.info("✅ FastAPI imported")

from fastapi.middleware.cors import CORSMiddleware
log.info("✅ CORSMiddleware imported")

from middleware.exception_handlers import catch_exception_middleware
log.info("✅ exception_handlers imported")

from routes.upload_pdf import router as upload_router
log.info("✅ upload_router imported")

from routes.ask_qus import router as ask_router
log.info("✅ ask_router imported")

from routes.evaluate import router as evaluate_router
log.info("✅ evaluate_router imported")

log.info("=== ALL IMPORTS DONE ===")

app = FastAPI(title="MediRAG Advanced API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.middleware("http")(catch_exception_middleware)

app.include_router(upload_router)
app.include_router(ask_router)
app.include_router(evaluate_router)

@app.get("/")
def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.exception_handlers import catch_exception_middleware

app = FastAPI(title="MediRAG Advanced API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.middleware("http")(catch_exception_middleware)

# ✅ Import routers lazily inside function
@app.on_event("startup")
async def startup():
    from routes.upload_pdf import router as upload_router
    from routes.ask_qus import router as ask_router
    from routes.evaluate import router as evaluate_router
    app.include_router(upload_router)
    app.include_router(ask_router)
    app.include_router(evaluate_router)

@app.get("/")
def health_check():
    return {"status": "running", "message": "MediRAG API is live!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)