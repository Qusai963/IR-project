from app.services.preprocessor.handlers.tokenization_handler import TokenizationHandler
from app.services.preprocessor.preprocessing_context import PreprocessingContext
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from .interfaces import ISearchStrategy
from app.models.search_models import SearchRequest
from fastapi import HTTPException
import numpy as np
from typing import List


class BM25SearchStrategy(ISearchStrategy):
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())

    def _tokenize(self, text: str) -> List[str]:
        processed_text = self.preprocessor.normalize(text)
        tokenizer = TokenizationHandler()
        context = PreprocessingContext(processed_text)
        processed_context = tokenizer.handle(context)
        return processed_context.tokens

    def search(self, request: SearchRequest) -> List[tuple]:
        assets = self.model_manager.get_model_assets(request.dataset_name, request.model)
        if not assets:
            raise HTTPException(
                status_code=503, 
                detail=f"Model '{request.model}' for dataset '{request.dataset_name}' not loaded."
            )
        raw_bm25_model = assets["model"]
        doc_ids = assets["doc_ids"]
        
        from app.services.bm25_service import BM25Service
        bm25_service = BM25Service()
        bm25_service.model = raw_bm25_model
        bm25_service.doc_ids = doc_ids
        
        scores = bm25_service.get_scores(request.query)
        
        top_indices = np.argsort(scores)[-request.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((doc_ids[idx], float(scores[idx])))
        
        return results 