from .interfaces import ISearchStrategy
from app.models.search_models import SearchRequest
from app.services.ranking_service import create_ranking_service
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from fastapi import HTTPException
import numpy as np
from typing import List

class TFIDFSearchStrategy(ISearchStrategy):
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())

    def search(self, request: SearchRequest) -> List[tuple]:
        assets = self.model_manager.get_model_assets(request.dataset_name, request.model)
        if not assets:
            raise HTTPException(
                status_code=503, 
                detail=f"Model '{request.model}' for dataset '{request.dataset_name}' not loaded."
            )
        
        tfidf_model = assets["model"]
        doc_ids = assets["doc_ids"]
        doc_vectors = assets["doc_vectors"]
        inverted_index = assets.get("inverted_index")
        
        if doc_vectors is None:
            raise HTTPException(status_code=500, detail="TF-IDF document vectors not found in model assets.")

        processed_query = self.preprocessor.normalize(request.query)
        query_terms = processed_query.split()
        candidate_indices = set()
        if inverted_index:
            for term in query_terms:
                candidate_indices.update(inverted_index.get(term, []))
        else:
            candidate_indices = set(range(len(doc_ids)))  

        if not candidate_indices:
            return []

        candidate_indices = list(candidate_indices)
        candidate_doc_vectors = doc_vectors[candidate_indices]
        candidate_doc_ids = [doc_ids[i] for i in candidate_indices]
        
        query_vector = tfidf_model.transform([request.query])
        
        ranking_service = create_ranking_service(request.model, candidate_doc_vectors, candidate_doc_ids)
        results = ranking_service.rank_documents(query_vector, top_k=request.top_k)
        
        return results 