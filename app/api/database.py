from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models.enums import DatasetName
from app.services.database_service import DatabaseService

router = APIRouter()

class GetDocumentsRequest(BaseModel):
    dataset_name: DatasetName
    doc_ids: List[str]

class GetDocumentsResponse(BaseModel):
    dataset_name: DatasetName
    documents: List[Dict[str, Any]]
    count: int

@router.post("/database/documents", response_model=GetDocumentsResponse)
def get_documents_by_ids(request: GetDocumentsRequest):
    """Get documents by their IDs from the database"""
    try:
        db_service = DatabaseService()
        documents = db_service.get_documents_by_ids(request.dataset_name, request.doc_ids)
        
        return GetDocumentsResponse(
            dataset_name=request.dataset_name,
            documents=documents,
            count=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")

@router.get("/database/documents/{dataset_name}")
def get_documents_by_dataset(dataset_name: DatasetName, limit: int = None):
    """Get documents from a dataset with optional limit"""
    try:
        db_service = DatabaseService()
        documents = db_service.get_docs_by_dataset(dataset_name, limit)
        
        return {
            "dataset_name": dataset_name,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")

@router.get("/database/health")
def database_health_check():
    """Health check for database service"""
    try:
        db_service = DatabaseService()
        test_docs = db_service.get_docs_by_dataset(DatasetName.ANTIQUE, limit=1)
        return {
            "status": "healthy",
            "service": "database",
            "connection": "active",
            "test_query": "successful"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "database",
            "error": str(e)
        } 