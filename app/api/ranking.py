from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Dict, Any
from app.models.enums import ModelType, DatasetName
from app.services.ranking_service import create_ranking_service
from app.services.model_manager import model_manager

router = APIRouter()

class RankRequest(BaseModel):
    query_vector: List[float]
    model_type: ModelType
    dataset_name: DatasetName
    top_k: int = 10

class RankResponse(BaseModel):
    model_type: ModelType
    dataset_name: DatasetName
    top_k: int
    results: List[Dict[str, Any]]
    success: bool

@router.post("/rank", response_model=RankResponse)
def rank_documents(request: RankRequest):
    """Rank documents using the specified model"""
    try:
        assets = model_manager.get_model_assets(request.dataset_name, request.model_type)
        if not assets:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {request.model_type} not found for dataset {request.dataset_name}"
            )
        
        import numpy as np
        if request.model_type == ModelType.TFIDF:
            from scipy.sparse import csr_matrix
            query_vector = csr_matrix(np.array(request.query_vector).reshape(1, -1))
        elif request.model_type == ModelType.BERT:
            query_vector = np.array(request.query_vector).reshape(1, -1)
        else:
            if len(request.query_vector) == 1 and isinstance(request.query_vector[0], str):
                query_text = request.query_vector[0]
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="For BM25, query_vector should contain the original query text as a single string"
                )
        
        if request.model_type == ModelType.TFIDF:
            doc_representations = assets["doc_vectors"]
        elif request.model_type == ModelType.BERT:
            doc_representations = assets["doc_embeddings"]
        else:
            doc_representations = None
        
        if request.model_type == ModelType.BM25:
            raw_bm25_model = assets["model"]
            doc_ids = assets["doc_ids"]
            
            from app.services.bm25_service import BM25Service
            bm25_service = BM25Service()
            bm25_service.model = raw_bm25_model
            bm25_service.doc_ids = doc_ids
            
            scores = bm25_service.get_scores(query_text)
            top_indices = np.argsort(scores)[-request.top_k:][::-1]
            results = [(doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        else:
            ranking_service = create_ranking_service(
                request.model_type, doc_representations, assets["doc_ids"]
            )
            results = ranking_service.rank_documents(query_vector, top_k=request.top_k)
        
        formatted_results = [
            {"doc_id": doc_id, "score": float(score)} 
            for doc_id, score in results
        ]
        
        print(f"Ranking service returning: {formatted_results}")
        
        return RankResponse(
            model_type=request.model_type,
            dataset_name=request.dataset_name,
            top_k=request.top_k,
            results=formatted_results,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {str(e)}")

@router.get("/rank/models/{dataset_name}")
def get_ranking_models(dataset_name: DatasetName):
    """Get available ranking models for a dataset"""
    try:
        available_models = {}
        for model_type in ModelType:
            try:
                assets = model_manager.get_model_assets(dataset_name, model_type)
                available_models[model_type.value] = assets is not None
            except:
                available_models[model_type.value] = False
        return {"dataset": dataset_name, "available_models": available_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}") 