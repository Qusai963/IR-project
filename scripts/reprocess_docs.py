import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.enums import DatasetName
from app.services.database_service import DatabaseService
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

def reprocess_data_for_dataset(dataset_name: DatasetName, limit: int = None):
    """
    Reprocesses the text of documents in the database for a specific dataset.
    """
    print(f"--- Starting data reprocessing for '{dataset_name}' ---")
    
    db_service = DatabaseService() 
    vsm_preprocessor = PreprocessingService(VSMPreprocessingStrategy())
    embedding_preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())
    
    docs_to_reprocess = db_service.get_docs_by_dataset(dataset_name, limit=limit)
    
    if not docs_to_reprocess:
        print(f"No documents found for dataset '{dataset_name}'")
        return
    
    for doc in docs_to_reprocess:
        vsm_text = vsm_preprocessor.normalize(doc['text'])
        embedding_text = embedding_preprocessor.normalize(doc['text'])
        
        db_service.update_processed_text(doc['doc_id'], dataset_name, vsm_text)
        db_service.update_processed_text(doc['doc_id'], dataset_name, embedding_text)
    
    print(f"--- Finished data reprocessing for '{dataset_name}' ---")
    print(f"   Documents reprocessed: {len(docs_to_reprocess)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reprocess document text in the database.")
    parser.add_argument("--dataset", type=str, default="antique", 
                       choices=["antique", "quora"],
                       help="Dataset name to reprocess")
    parser.add_argument("--limit", type=int, help="Limit number of documents to reprocess")
    
    args = parser.parse_args()
    
    dataset_name = DatasetName(args.dataset)
    reprocess_data_for_dataset(dataset_name, args.limit) 