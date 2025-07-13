from .interfaces import ISearchStrategy
from app.models.search_models import SearchRequest
from app.services.ranking_service import create_ranking_service
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.models.enums import ModelType
from fastapi import HTTPException
from typing import List

class BERTSearchStrategy(ISearchStrategy):
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())

    def search(self, request: SearchRequest) -> List[tuple]:
        assets = self.model_manager.get_model_assets(request.dataset_name, request.model)
        if not assets:
            raise HTTPException(
                status_code=503, 
                detail=f"Model '{request.model}' for dataset '{request.dataset_name}' not loaded."
            )
        
        processed_query = self.preprocessor.normalize(request.query)
        
        query_vector = assets["model"].encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        doc_representations = assets["doc_embeddings"]
        if doc_representations is None:
            raise HTTPException(status_code=500, detail="BERT document embeddings not found in model assets.")
        
        ranking_service = create_ranking_service(request.model, doc_representations, assets["doc_ids"])
        results = ranking_service.rank_documents(query_vector, top_k=request.top_k)
        
        return results 