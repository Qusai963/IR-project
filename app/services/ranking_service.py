from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Union
from scipy.sparse import csr_matrix
import numpy as np
from abc import ABC, abstractmethod
from app.models.enums import ModelType
from app.services.bm25_service import BM25Service

class BaseRankingService(ABC):
    
    def __init__(self, doc_ids: List[str]):
        self.doc_ids = doc_ids
    
    @abstractmethod
    def rank_documents(self, query_representation, top_k: int = 5) -> List[Tuple[str, float]]:
        pass

class TfidfRankingService(BaseRankingService):
    def __init__(self, doc_vectors: Union[csr_matrix, np.ndarray], doc_ids: List[str]):
        super().__init__(doc_ids)
        self.doc_vectors = doc_vectors

    def rank_documents(self, query_vector: Union[csr_matrix, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.doc_vectors is None:
            raise ValueError("TfidfRankingService has not been initialized with document vectors.")

        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        ranked_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.doc_ids[i], float(similarities[i])) for i in ranked_indices if similarities[i] > 0]
        return results

class BertRankingService(BaseRankingService):
    def __init__(self, doc_embeddings: np.ndarray, doc_ids: List[str]):
        super().__init__(doc_ids)
        self.doc_embeddings = doc_embeddings

    def rank_documents(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.doc_embeddings is None:
            raise ValueError("BertRankingService has not been initialized with document embeddings.")

        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()
        ranked_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(self.doc_ids[i], float(similarities[i])) for i in ranked_indices if similarities[i] > 0]
        return results

class BM25RankingService(BaseRankingService):
    def __init__(self, bm25_model: BM25Service, doc_ids: List[str]):
        super().__init__(doc_ids)
        self.bm25_model = bm25_model

    def rank_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.bm25_model is None:
            raise ValueError("BM25RankingService has not been initialized with BM25 model.")
        
        return self.bm25_model.rank_documents(query, top_k)

def create_ranking_service(model_type: ModelType, doc_representations, doc_ids: List[str]) -> BaseRankingService:
    if model_type == ModelType.TFIDF:
        return TfidfRankingService(doc_representations, doc_ids)
    elif model_type == ModelType.BERT:
        return BertRankingService(doc_representations, doc_ids)
    elif model_type == ModelType.BM25:
        return BM25RankingService(doc_representations, doc_ids)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

class RankingService(BaseRankingService):
    
    def __init__(self, doc_vectors: Union[csr_matrix, np.ndarray], doc_ids: List[str]):
        super().__init__(doc_ids)
        self.doc_vectors = doc_vectors

    def rank_documents(self, query_vector: Union[csr_matrix, np.ndarray], top_k: int = 5) -> List[Tuple[str, float]]:
        if self.doc_vectors is None:
            raise ValueError("RankingService has not been initialized with document vectors.")

        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        if top_k > self.doc_vectors.shape[0]:
            top_k = self.doc_vectors.shape[0]
        
        ranked_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [(self.doc_ids[i], float(similarities[i])) for i in ranked_indices if similarities[i] > 0]
        
        return results