import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.enums import DatasetName
from app.services.dataset_loader import DatasetService
from app.services.database_service import DatabaseService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

def ingest_data_for_dataset(dataset_name: DatasetName, limit: int = None):

    print(f"--- Starting data ingestion for '{dataset_name}' ---")
    
    dataset_service = DatasetService()
    db_service = DatabaseService() 
    
    db_service.create_documents_table()

    raw_docs = dataset_service.get_docs(dataset_name, limit=limit)
    
    if not raw_docs:
        print(f"No documents found for dataset '{dataset_name}'")
        return
    
    vsm_preprocessor = PreprocessingService(VSMPreprocessingStrategy())
    embedding_preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())
    
    docs_to_insert = []
    for doc in raw_docs:
        vsm_text = vsm_preprocessor.normalize(doc.text)
        embedding_text = embedding_preprocessor.normalize(doc.text)
        docs_to_insert.append({
            "doc_id": doc.doc_id,
            "text": doc.text,
            "vsm_text": vsm_text,
            "embedding_text": embedding_text
        })
    
    db_service.insert_docs(docs_to_insert, dataset_name)
    
    print(f"--- Finished data ingestion for '{dataset_name}' ---")
    print(f"   Documents processed: {len(raw_docs)}")
    print(f"   Documents stored: {len(docs_to_insert)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest documents to database")
    parser.add_argument("--dataset", type=str, default="antique", 
                       choices=["antique", "quora"],
                       help="Dataset name to ingest")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    dataset_name = DatasetName(args.dataset)
    ingest_data_for_dataset(dataset_name, args.limit)