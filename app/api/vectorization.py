from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from app.models.enums import ModelType, DatasetName
from app.services.vectorization_service import VectorizationService
from app.services.model_manager import model_manager

router = APIRouter()

class VectorizeRequest(BaseModel):
    query: str
    model_type: ModelType
    dataset_name: DatasetName

class VectorizeResponse(BaseModel):
    query: str
    model_type: ModelType
    dataset_name: DatasetName
    query_vector: List[float]
    doc_representations_shape: List[int]
    success: bool

@router.post("/vectorize", response_model=VectorizeResponse)
def vectorize_query(request: VectorizeRequest):
    """Vectorize a query using the specified model"""
    try:
        assets = model_manager.get_model_assets(request.dataset_name, request.model_type)
        if not assets:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {request.model_type} not found for dataset {request.dataset_name}"
            )
        
        vectorization_service = VectorizationService()
        query_vector, doc_representations = vectorization_service.vectorize_query(
            request.query, request.model_type, assets
        )
        
        if hasattr(query_vector, 'toarray'):
            query_vector_list = query_vector.toarray().flatten().tolist()
        elif hasattr(query_vector, 'flatten'):
            query_vector_list = query_vector.flatten().tolist()
        else:
            query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else str(query_vector)
        
        return VectorizeResponse(
            query=request.query,
            model_type=request.model_type,
            dataset_name=request.dataset_name,
            query_vector=query_vector_list,
            doc_representations_shape=list(doc_representations.shape) if hasattr(doc_representations, 'shape') else [],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

@router.get("/vectorize/models/{dataset_name}")
def get_available_models(dataset_name: DatasetName):
    """Get available models for a dataset"""
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