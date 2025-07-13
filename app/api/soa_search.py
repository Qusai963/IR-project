from fastapi import APIRouter, HTTPException
from app.models.search_models import SearchRequest, SearchResponse
from app.services.soa_search_service import SOASearchService
import traceback

router = APIRouter()

soa_search_service = SOASearchService()

@router.post("/soa-search", response_model=SearchResponse)
def soa_search(request: SearchRequest):
    """
    SOA-compliant search endpoint that communicates with other services via HTTP APIs.
    This demonstrates proper service isolation and communication.
    """
    try:
        return soa_search_service.search(request)
    except HTTPException:
        raise
    except Exception as e:
        print(f"An error occurred during SOA search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during SOA search")

@router.get("/soa-search/health")
def soa_search_health():
    """Health check for SOA search service and its dependencies"""
    return soa_search_service.health_check()

@router.get("/soa-search/services")
def list_soa_services():
    """List all services in the SOA architecture"""
    return {
        "services": [
            {
                "name": "preprocessing",
                "endpoint": "/api/preprocess",
                "description": "Text preprocessing service",
                "methods": ["POST"]
            },
            {
                "name": "vectorization", 
                "endpoint": "/api/vectorize",
                "description": "Query/document vectorization service",
                "methods": ["POST"]
            },
            {
                "name": "ranking",
                "endpoint": "/api/rank", 
                "description": "Document ranking service",
                "methods": ["POST"]
            },
            {
                "name": "database",
                "endpoint": "/api/database/documents",
                "description": "Document storage and retrieval service", 
                "methods": ["POST", "GET"]
            },
            {
                "name": "search",
                "endpoint": "/api/soa-search",
                "description": "Main search service (orchestrates other services)",
                "methods": ["POST"]
            }
        ],
        "architecture": "SOA (Service-Oriented Architecture)",
        "communication": "HTTP APIs between services",
        "complete_soa": "All services communicate via APIs only"
    } 