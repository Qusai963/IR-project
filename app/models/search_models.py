from pydantic import BaseModel
from typing import List
from app.models.enums import DatasetName, ModelType

class SearchRequest(BaseModel):
    dataset_name: DatasetName
    query: str
    model: ModelType
    top_k: int = 5
    weights: List[float] = [0.5, 0.5]

class SearchResult(BaseModel):
    doc_id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    processed_query: str
    model: ModelType
    results: List[SearchResult]
    weights: List[float] = None