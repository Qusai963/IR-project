from fastapi import APIRouter, HTTPException
from app.models.enums import DatasetName
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.dataset_loader import DatasetService

router = APIRouter()
preprocessor = PreprocessingService(VSMPreprocessingStrategy())
dataset_service = DatasetService()

@router.get("/qrels/{dataset_name}")
def get_qrels(dataset_name: DatasetName, limit: int = None):
    qrels = dataset_service.get_qrels(dataset_name, limit)
    if not qrels:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return [{"query_id": q.query_id, "doc_id": q.doc_id, "relevance": q.relevance} for q in qrels]