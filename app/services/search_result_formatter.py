from app.models.search_models import SearchResult
from typing import List, Dict

class SearchResultFormatter:
    def __init__(self, database_service):
        self.database_service = database_service

    def get_document_texts(self, dataset_name, results: List[tuple]) -> Dict[str, str]:
        doc_ids = [doc_id for doc_id, _ in results]
        documents = self.database_service.get_documents_by_ids(dataset_name, doc_ids)
        return {doc['id']: doc['text'] for doc in documents}

    def format_results(self, results: List[tuple], doc_text_map: Dict[str, str]) -> List[SearchResult]:
        return [
            SearchResult(
                doc_id=doc_id,
                score=score,
                text=doc_text_map.get(doc_id, "Document not found")
            )
            for doc_id, score in results
        ] 