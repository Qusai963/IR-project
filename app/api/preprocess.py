from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.enums import DatasetName
from app.services.dataset_loader import DatasetService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

router = APIRouter()

class PreprocessRequest(BaseModel):
    text: str
    strategy: str = "vsm"

preprocessor_vsm = PreprocessingService(VSMPreprocessingStrategy())
preprocessor_embedding = PreprocessingService(EmbeddingPreprocessingStrategy())
dataset_service = DatasetService()

@router.post("/preprocess")
def preprocess(request: PreprocessRequest):
    """Preprocess text using specified strategy"""
    try:
        if request.strategy == "vsm":
            processed_text = preprocessor_vsm.normalize(request.text)
        elif request.strategy == "embedding":
            processed_text = preprocessor_embedding.normalize(request.text)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        return {
            "original": request.text,
            "processed": processed_text,
            "strategy": request.strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.get("/preprocess/{dataset_name}")
def preprocess_get(text: str, dataset_name: DatasetName):
    """Legacy GET endpoint for backward compatibility"""
    return [{"original": text, "processed": preprocessor_vsm.normalize(text)}]