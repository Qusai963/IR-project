from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.query_refinement_service import QueryRefinementService

router = APIRouter()
query_refinement_service = QueryRefinementService()

class QueryRefinementRequest(BaseModel):
    query: str

@router.post("/query-refinement")
def get_query_suggestions(request: QueryRefinementRequest):
    try:
        suggestions = query_refinement_service.suggest_improvements(request.query)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 