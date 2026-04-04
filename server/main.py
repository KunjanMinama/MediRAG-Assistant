# ✅ Don't import heavy modules at top level
# They will be imported only when routes are called

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