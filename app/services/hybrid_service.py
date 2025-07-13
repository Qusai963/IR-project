from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.models.enums import ModelType
from app.services.interfaces import IVectorizationService
from app.services.ranking_service import create_ranking_service

class HybridService:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        scores_array = np.array(scores).reshape(-1, 1)
        normalized = self.scaler.fit_transform(scores_array)
        return normalized.flatten().tolist()
    
    def _combine_scores_parallel(self, tfidf_scores: List[float], bert_scores: List[float], bm25_scores: List[float] = None,
                                weights: Tuple[float, float, float] = (1/3, 1/3, 1/3),
                                doc_ids: List[str] = None, top_k: int = None) -> List[Tuple[str, float]]:
        """Combine scores from parallel models using weighted fusion (supports 2 or 3 models). For 3 models, first return intersection, then fill with highest scores."""
        if bm25_scores is not None and doc_ids is not None and top_k is not None:
            tfidf_norm = self._normalize_scores(tfidf_scores)
            bert_norm = self._normalize_scores(bert_scores)
            bm25_norm = self._normalize_scores(bm25_scores)
            w1, w2, w3 = weights
            doc_tuples = []
            for i, doc_id in enumerate(doc_ids):
                t, b, m = tfidf_norm[i], bert_norm[i], bm25_norm[i]
                weighted = w1 * t + w2 * b + w3 * m
                doc_tuples.append((doc_id, t, b, m, weighted))
            intersection = [d for d in doc_tuples if d[1] > 0 and d[2] > 0 and d[3] > 0]
            intersection.sort(key=lambda x: x[4], reverse=True)
            results = [(d[0], d[4]) for d in intersection]
            if len(results) < top_k:
                rest = [d for d in doc_tuples if not (d[1] > 0 and d[2] > 0 and d[3] > 0)]
                rest.sort(key=lambda x: x[4], reverse=True)
                for d in rest:
                    if len(results) >= top_k:
                        break
                    results.append((d[0], d[4]))
            return results[:top_k]
        else:
            if len(tfidf_scores) != len(bert_scores):
                raise ValueError("Score lists must have the same length")
            tfidf_norm = self._normalize_scores(tfidf_scores)
            bert_norm = self._normalize_scores(bert_scores)
            w1, w2 = weights[:2]
            combined = [w1 * t + w2 * b for t, b in zip(tfidf_norm, bert_norm)]
            return list(zip(doc_ids, combined)) if doc_ids is not None else combined
    
    def search_parallel(self, query: str, tfidf_assets: Dict[str, Any], bert_assets: Dict[str, Any], bm25_assets: Dict[str, Any] = None,
                       weights: Tuple[float, float, float] = (1/3, 1/3, 1/3), top_k: int = 5) -> List[Tuple[str, float]]:
        """Search using parallel hybrid approach with weighted fusion (supports 2 or 3 models)"""
        from app.services.vectorization_service import VectorizationService
        vectorization_service = VectorizationService()
        tfidf_query_vector, tfidf_doc_vectors = vectorization_service.vectorize_query(
            query, ModelType.TFIDF, tfidf_assets
        )
        bert_query_vector, bert_doc_vectors = vectorization_service.vectorize_query(
            query, ModelType.BERT, bert_assets
        )
        tfidf_ranking = create_ranking_service(ModelType.TFIDF, tfidf_doc_vectors, tfidf_assets["doc_ids"])
        bert_ranking = create_ranking_service(ModelType.BERT, bert_doc_vectors, bert_assets["doc_ids"])
        tfidf_results = tfidf_ranking.rank_documents(tfidf_query_vector, top_k=len(tfidf_assets["doc_ids"]))
        bert_results = bert_ranking.rank_documents(bert_query_vector, top_k=len(bert_assets["doc_ids"]))
        tfidf_scores = {doc_id: score for doc_id, score in tfidf_results}
        bert_scores = {doc_id: score for doc_id, score in bert_results}
        all_doc_ids = list(tfidf_assets["doc_ids"])
        tfidf_score_list = [tfidf_scores.get(doc_id, 0) for doc_id in all_doc_ids]
        bert_score_list = [bert_scores.get(doc_id, 0) for doc_id in all_doc_ids]
        if bm25_assets is not None:
            from app.services.bm25_service import BM25Service
            bm25_service = BM25Service()
            bm25_service.model = bm25_assets["model"]
            bm25_service.doc_ids = bm25_assets["doc_ids"]
            bm25_scores_dict = dict(bm25_service.rank_documents(query, top_k=len(bm25_assets["doc_ids"])))
            bm25_score_list = [bm25_scores_dict.get(doc_id, 0) for doc_id in all_doc_ids]
            combined_results = self._combine_scores_parallel(tfidf_score_list, bert_score_list, bm25_score_list, weights, all_doc_ids, top_k)
        else:
            combined_results = self._combine_scores_parallel(tfidf_score_list, bert_score_list, None, weights, all_doc_ids, top_k)
        return combined_results 