from fastapi import APIRouter, HTTPException, Query
from app.models.enums import DatasetName
from app.services.dataset_loader import DatasetService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.exporter import ExportService

router = APIRouter()
preprocessor = PreprocessingService(VSMPreprocessingStrategy())
dataset_service = DatasetService()
exporter = ExportService()

@router.get("/export/{dataset_name}")
def export_processed_data(
    dataset_name: DatasetName,
    data_type: str = Query("queries", enum=["queries", "docs"]),
    format: str = Query("csv", enum=["csv", "jsonl"]),
    limit: int = 10
):
    if data_type == "queries":
        items = dataset_service.get_queries(dataset_name, limit)
        data = [{"id": q.query_id, "original": q.text, "processed": preprocessor.normalize(q.text)} for q in items]
    elif data_type == "docs":
        items = dataset_service.get_docs(dataset_name, limit)
        data = [{"id": d.doc_id, "original": d.text, "processed": preprocessor.normalize(d.text)} for d in items]
    else:
        raise HTTPException(status_code=400, detail="Invalid data_type")

    filename = f"{dataset_name}_{data_type}_processed"

    if format == "csv":
        filepath = exporter.export_to_csv(data, filename)
    else:
        filepath = exporter.export_to_jsonl(data, filename)

    return {"message": f"Data exported successfully", "file": filepath}