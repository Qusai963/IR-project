import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from app.api import search, queries, qrels, export, query_refinement, advanced_features, preprocess, vectorization, ranking, soa_search, database
from app.services.model_manager import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_all_models()
    yield

app = FastAPI(title="IR Project API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocess.router, prefix="/api", tags=["preprocessing"])
app.include_router(vectorization.router, prefix="/api", tags=["vectorization"])
app.include_router(ranking.router, prefix="/api", tags=["ranking"])
app.include_router(database.router, prefix="/api", tags=["database"])

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(soa_search.router, prefix="/api", tags=["soa_search"])
app.include_router(advanced_features.router, prefix="/api", tags=["advanced_features"])

app.include_router(queries.router, prefix="/api", tags=["queries"])
app.include_router(qrels.router, prefix="/api", tags=["qrels"])
app.include_router(export.router, prefix="/api", tags=["export"])

app.include_router(query_refinement.router, prefix="/api", tags=["query_refinement"])

@app.get("/api/models/status")
async def get_models_status():
    return model_manager.get_loaded_models()

@app.get("/api/health")
async def health_check():
    """Health check endpoint for all services"""
    return {
        "status": "healthy",
        "services": {
            "preprocessing": "available",
            "vectorization": "available", 
            "ranking": "available",
            "database": "available",
            "search": "available",
            "soa_search": "available",
            "model_manager": "available"
        }
    }

@app.get("/health")
async def root_health_check():
    """Root health check endpoint"""
    return {
        "status": "healthy",
        "message": "IR Project API is running",
        "version": "1.0.0"
    }

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)