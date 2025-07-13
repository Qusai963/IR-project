from fastapi import APIRouter, HTTPException
from app.models.search_models import SearchRequest, SearchResult, SearchResponse
from app.models.enums import DatasetName, ModelType
from app.services.model_manager import model_manager
from app.services.database_service import DatabaseService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.vectorization_service import VectorizationService
from app.services.search_strategies.bm25_strategy import BM25SearchStrategy
from app.services.search_strategies.hybrid_strategy import HybridSearchStrategy
from app.services.search_strategies.tfidf_strategy import TFIDFSearchStrategy
from app.services.search_strategies.bert_strategy import BERTSearchStrategy
from app.services.search_result_formatter import SearchResultFormatter
from app.services.search_service import SearchService
from app.services.ranking_service import create_ranking_service
import traceback

router = APIRouter()

preprocessing_service = PreprocessingService(VSMPreprocessingStrategy())
db_service = DatabaseService()
vectorization_service = VectorizationService()
result_formatter = SearchResultFormatter(db_service)

strategies = {
    ModelType.BM25: BM25SearchStrategy(model_manager),
    ModelType.TFIDF: TFIDFSearchStrategy(model_manager),
    ModelType.BERT: BERTSearchStrategy(model_manager),
    ModelType.HYBRID_PARALLEL: HybridSearchStrategy(model_manager),
}

search_service = SearchService(
    strategies,
    result_formatter,
    preprocessing_service,
    vectorization_service,
    model_manager,
    db_service,
    create_ranking_service
)

@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        return search_service.search(request)
    except HTTPException:
        raise
    except Exception as e:
        print(f"An error occurred during search: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error during search")