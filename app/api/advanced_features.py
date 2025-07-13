from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from app.models.enums import DatasetName, ModelType
from app.services.database_service import DatabaseService
from app.services.model_manager import model_manager
from app.services.search_strategies.hybrid_strategy import HybridSearchStrategy
from app.models.search_models import SearchRequest
from app.models.enums import ModelType
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.dataset_loader import DatasetService

router = APIRouter()
vsm_preprocessor = PreprocessingService(VSMPreprocessingStrategy())
embedding_preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())
dataset_service = DatasetService()

class HybridSearchRequest(BaseModel):
    dataset_name: DatasetName
    query: str
    weights: Optional[List[float]] = None
    top_k: int = 5

class HybridSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

class RAGResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    context_used: int

@router.post("/hybrid/search", response_model=HybridSearchResponse)
def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search using the new HybridSearchStrategy.
    
    Weights format: [tfidf_weight, bert_weight, bm25_weight]
    - Use 0 to exclude a model (e.g., [0.5, 0.5, 0] for TF-IDF + BERT only)
    - Weights are automatically normalized to sum to 1
    - If no weights provided, equal weights are used for available models
    """
    try:
        hybrid_strategy = HybridSearchStrategy(model_manager)
        
        search_request = SearchRequest(
            dataset_name=request.dataset_name,
            query=request.query,
            model=ModelType.HYBRID_PARALLEL,
            top_k=request.top_k,
            weights=request.weights
        )
        
        results = hybrid_strategy.search(search_request)
        
        db_service = DatabaseService()
        doc_ids_list = [doc_id for doc_id, _ in results]
        docs = db_service.get_documents_by_ids(request.dataset_name, doc_ids_list)
        doc_texts = {doc['id']: doc['text'] for doc in docs}
        
        formatted_results = []
        for doc_id, score in results:
            formatted_results.append({
                "doc_id": doc_id,
                "score": score,
                "text": doc_texts.get(doc_id, "Document not found")
            })
        
        return HybridSearchResponse(
            query=request.query,
            results=formatted_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in hybrid search: {str(e)}")

@router.post("/rag/search", response_model=RAGResponse)
def rag_search(request: RAGRequest):
    """Perform RAG search"""
    try:
        rag_service = model_manager.get_rag_service()
        
        if rag_service.index is None:
            raise HTTPException(status_code=503, detail="Vector store not loaded")
        
        results = rag_service.rag_search(request.query, request.top_k)
        
        return RAGResponse(**results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in RAG search: {str(e)}")

@router.post("/rag/initialize")
def initialize_rag(dataset_name: DatasetName):
    """Initialize RAG using existing BERT embeddings"""
    try:
        rag_service = model_manager.get_rag_service()
        
        results = rag_service.initialize_from_dataset(dataset_name)
        return {
            "message": "RAG initialized successfully using existing BERT embeddings", 
            "details": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error initializing RAG: {str(e)}. Ensure BERT model is trained for this dataset."
        )

@router.get("/rag/status")
def get_rag_status():
    """Get RAG service status"""
    try:
        rag_service = model_manager.get_rag_service()
        return rag_service.get_vector_store_info()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting RAG status: {str(e)}") 