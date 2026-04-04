###main.py file
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middleware.exception_handlers import catch_exception_middleware
from routes.upload_pdf import router as upload_router
from routes.ask_qus import router as ask_router
from routes.evaluate import router as evaluate_router
import logging
logging.basicConfig(level=logging.INFO)

app=FastAPI(title="Medical AssistantAPI",description="API for AI Assistant Chatbot")

@app.get("/")
def health_check():
    return {"status": "running", "message": "MediRAG API is live!"}

@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI startup complete!")




# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# middleware exception handlers
app.middleware("http")(catch_exception_middleware)


# routers

#1. upload pdfs documents
app.include_router(upload_router)
#2. asking query
app.include_router(ask_router)
#3. evaluation
app.include_router(evaluate_router)





# ✅ Add this at the bottom
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
