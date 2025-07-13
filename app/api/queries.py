from fastapi import APIRouter, HTTPException
from app.models.enums import DatasetName
from app.services.dataset_loader import DatasetService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

router = APIRouter()
preprocessor = PreprocessingService(VSMPreprocessingStrategy())
dataset_service = DatasetService()

@router.get("/queries/{dataset_name}")
def get_queries(dataset_name: DatasetName, limit: int = None):
    queries = dataset_service.get_queries(dataset_name, limit)
    if not queries:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return [{"original": q.text, "processed": preprocessor.normalize(q.text)} for q in queries]