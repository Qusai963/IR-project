from typing import List
from app.models.search_models import SearchRequest

class ISearchStrategy:
    def search(self, request: SearchRequest) -> List[tuple]:
        raise NotImplementedError 