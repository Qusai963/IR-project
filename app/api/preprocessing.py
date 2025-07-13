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

class PreprocessResponse(BaseModel):
    original: str
    processed: str
    strategy: str

@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text(request: PreprocessRequest):
    """Preprocess text using the specified strategy"""
    try:
        if request.strategy == "vsm":
            preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        elif request.strategy == "embedding":
            preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())
        else:
            raise HTTPException(status_code=400, detail="Invalid strategy. Use 'vsm' or 'embedding'")
        
        processed_text = preprocessor.normalize(request.text)
        return PreprocessResponse(
            original=request.text,
            processed=processed_text,
            strategy=request.strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@router.get("/preprocess/{dataset_name}")
def preprocess_dataset(dataset_name: DatasetName, limit: int = None):
    """Get preprocessed documents from a dataset"""
    try:
        dataset_service = DatasetService()
        preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        docs = dataset_service.get_docs(dataset_name, limit)
        if not docs:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return [{"doc_id": d.doc_id, "original": d.text, "processed": preprocessor.normalize(d.text)} for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset preprocessing failed: {str(e)}") 