from fastapi import APIRouter, HTTPException
from app.models.enums import DatasetName
from app.services.dataset_loader import DatasetService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

router = APIRouter()
preprocessor = PreprocessingService(VSMPreprocessingStrategy())
dataset_service = DatasetService()

@router.get("/docs/{dataset_name}")
def get_docs(dataset_name: DatasetName, limit: int = None):
    docs = dataset_service.get_docs(dataset_name, limit)
    if not docs:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return [{"doc_id": d.doc_id, "original": d.text, "processed": preprocessor.normalize(d.text)} for d in docs]