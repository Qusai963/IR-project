from typing import Dict, Any, Tuple
from app.models.enums import ModelType
from app.services.interfaces import IVectorizationService

class VectorizationService(IVectorizationService):
    
    def vectorize_query(self, query: str, model_type: ModelType, assets: Dict[str, Any]) -> Tuple[Any, Any]:
        if model_type == ModelType.TFIDF:
            return self._vectorize_query_tfidf(query, assets)
        elif model_type == ModelType.BM25:
            return self._vectorize_query_bm25(query, assets)
        else:
            return self._vectorize_query_bert(query, assets)
    
    def _vectorize_query_tfidf(self, query: str, assets: Dict[str, Any]) -> Tuple[Any, Any]:
        vectorizer = assets["vectorizer"]
        query_vector = vectorizer.transform([query])
        doc_representations = assets["doc_vectors"]
        return query_vector, doc_representations
    
    def _vectorize_query_bm25(self, query: str, assets: Dict[str, Any]) -> Tuple[Any, Any]:
        from app.services.bm25_service import BM25Service
        bm25_service = BM25Service()
        bm25_service.model = assets["model"]
        bm25_service.doc_ids = assets["doc_ids"]
        return query, bm25_service
    
    def _vectorize_query_bert(self, query: str, assets: Dict[str, Any]) -> Tuple[Any, Any]:
        model = assets["model"]
        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        doc_representations = assets["doc_embeddings"]
        return query_vector, doc_representations 