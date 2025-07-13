import os
import sys
import joblib
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.bm25_service import BM25Service
from app.services.database_service import DatabaseService
from app.services.tfidf_service import TfidfService 
from app.services.bert_service import BertService
from app.models.enums import DatasetName
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

def train_tfidf_model(dataset_name: DatasetName, doc_ids: list, raw_texts: list):
    print("Fitting TF-IDF model...")
    tfidf_service = TfidfService()
    doc_vectors, fitted_vectorizer = tfidf_service.fit_transform(raw_texts)
    print("TF-IDF model fitted successfully.")

    print("Building inverted index...")
    vocabulary = fitted_vectorizer.vocabulary_
    inverted_index = {term: [] for term in vocabulary}
    for term, col_idx in vocabulary.items():
        doc_indices = doc_vectors[:, col_idx].nonzero()[0]
        inverted_index[term] = doc_indices.tolist()
    print("Inverted index built.")

    output_dir = f"app/saved_models/{dataset_name}/tfidf"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(fitted_vectorizer, os.path.join(output_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(doc_vectors, os.path.join(output_dir, "tfidf_vectors.joblib"))
    joblib.dump(inverted_index, os.path.join(output_dir, "inverted_index.joblib"))
    
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)
    
    print(f"TF-IDF model artifacts saved successfully to {output_dir}")

def train_bert_model(dataset_name: DatasetName, doc_ids: list, embedding_texts: list):
    print("Fitting BERT model...")
    bert_service = BertService()
    bert_embeddings, bert_model = bert_service.fit_transform(embedding_texts)
    print("BERT model fitted successfully.")

    output_dir = f"app/saved_models/{dataset_name}/bert"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(bert_model, os.path.join(output_dir, "bert_model.joblib"))
    joblib.dump(bert_embeddings, os.path.join(output_dir, "bert_embeddings.joblib"))
    
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)
    

    print(f"BERT model artifacts saved successfully to {output_dir}")

def train_bm25_model(dataset_name: DatasetName, doc_ids: list, vsm_texts: list):
    print(f"--- Starting BM25 training for dataset: {dataset_name} ---")
        
    print("Fitting BM25 model...")
    bm25_service = BM25Service()
    bm25_model, doc_ids = bm25_service.fit_transform(vsm_texts, doc_ids)
    print("BM25 model fitted successfully.")

    output_dir = f"app/saved_models/{dataset_name}/bm25"
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(bm25_model, os.path.join(output_dir, "bm25_model.joblib"))
    
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)
    
    print(f"BM25 model artifacts saved successfully to {output_dir}")


def train_and_save_model(dataset_name: DatasetName, model_type: str = "both", limit: int = None):

    print(f"--- Starting training for dataset: {dataset_name} ---")
    db_service = DatabaseService()
    
    doc_ids, raw_texts, vsm_texts, embedding_texts = db_service.get_processed_docs_by_dataset(dataset_name, limit=limit)
    if not doc_ids:
        print(f"No documents in DB for {dataset_name}. Run ingest_to_db.py or use the API endpoint /api/documents/process")
        return

  
    if model_type in ["tfidf", "both"]:
        train_tfidf_model(dataset_name, doc_ids, raw_texts)
    
    if model_type in ["bert", "both"]:
        train_bert_model(dataset_name, doc_ids, embedding_texts)

    if model_type in ["bm25", "both"]:
        train_bm25_model(dataset_name, doc_ids, vsm_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF and/or BERT models")
    parser.add_argument("--dataset", type=str, default="antique", help="Dataset name to train on")
    parser.add_argument("--model", type=str, choices=["tfidf", "bert", "bm25", "both"], default="both", 
                       help="Model type to train (tfidf, bert, or both)")
    parser.add_argument("--limit", type=int, help="Limit number of documents to process")
    
    args = parser.parse_args()
    
    train_and_save_model(args.dataset, args.model, args.limit)