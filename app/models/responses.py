from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[SearchResult]