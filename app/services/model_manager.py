import os
import joblib
import json
from typing import Dict, Any, Optional
from app.models.enums import DatasetName, ModelType
from app.services.interfaces import IModelManager
from app.services.bm25_service import BM25Service
from app.services.rag_service import RAGService

class ModelManager(IModelManager):
    def __init__(self):
        self.model_cache: Dict[str, Dict[str, Any]] = {}
        self.app_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.saved_models_dir = os.path.join(self.app_root_dir, "saved_models")
        self.rag_service = RAGService()
        
    def load_all_models(self):
        print("Loading all model assets at startup...")
        
        for dataset_name in DatasetName:
            for model_type in ModelType:
                try:
                    self._load_model_assets(dataset_name, model_type)
                    print(f"✓ Loaded {model_type} model for {dataset_name}")
                except Exception as e:
                    print(f"✗ Failed to load {model_type} model for {dataset_name}: {e}")
        
        try:
            if self.rag_service.load_vector_store():
                print("✓ Loaded RAG metadata")
            else:
                print("⚠ RAG not initialized - use /api/rag/initialize endpoint")
        except Exception as e:
            print(f"✗ Failed to load RAG metadata: {e}")
        
        print(f"Model loading complete. Loaded {len(self.model_cache)} models.")
    
    def _load_model_assets(self, dataset_name: DatasetName, model_type: ModelType):
        cache_key = f"{dataset_name}_{model_type}"
        
        if model_type == ModelType.TFIDF:
            model_path = os.path.join(self.saved_models_dir, dataset_name, "tfidf")
        elif model_type == ModelType.BM25:
            model_path = os.path.join(self.saved_models_dir, dataset_name, "bm25")
        else:
            model_path = os.path.join(self.saved_models_dir, dataset_name, "bert")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        try:
            if model_type == ModelType.TFIDF:
                vectorizer = joblib.load(os.path.join(model_path, "tfidf_vectorizer.joblib"))
                doc_vectors = joblib.load(os.path.join(model_path, "tfidf_vectors.joblib"))
                inverted_index_path = os.path.join(model_path, "inverted_index.joblib")
                if os.path.exists(inverted_index_path):
                    inverted_index = joblib.load(inverted_index_path)
                else:
                    inverted_index = None
                with open(os.path.join(model_path, "doc_ids.json"), "r") as f:
                    doc_ids = json.load(f)
                model = vectorizer
                doc_embeddings = None
            elif model_type == ModelType.BM25:
                bm25_model = joblib.load(os.path.join(model_path, "bm25_model.joblib"))
                with open(os.path.join(model_path, "doc_ids.json"), "r") as f:
                    doc_ids = json.load(f)
                model = bm25_model
                vectorizer = None
                doc_vectors = None
                doc_embeddings = None
            else:
                model = joblib.load(os.path.join(model_path, "bert_model.joblib"))
                doc_embeddings = joblib.load(os.path.join(model_path, "bert_embeddings.joblib"))
                with open(os.path.join(model_path, "doc_ids.json"), "r") as f:
                    doc_ids = json.load(f)
                vectorizer = None
                doc_vectors = None

            self.model_cache[cache_key] = {
                "model": model,
                "vectorizer": vectorizer,
                "doc_vectors": doc_vectors,
                "doc_embeddings": doc_embeddings,
                "doc_ids": doc_ids,
                "inverted_index": inverted_index if model_type == ModelType.TFIDF else None
            }
            
        except Exception as e:
            raise Exception(f"Failed to load model assets: {e}")
    
    def get_model_assets(self, dataset_name: DatasetName, model_type: ModelType) -> Optional[Dict[str, Any]]:
        cache_key = f"{dataset_name}_{model_type}"
        return self.model_cache.get(cache_key)
    
    def is_model_loaded(self, dataset_name: DatasetName, model_type: ModelType) -> bool:
        cache_key = f"{dataset_name}_{model_type}"
        return cache_key in self.model_cache
    
    def get_loaded_models(self) -> Dict[str, bool]:
        loaded_models = {}
        for dataset_name in DatasetName:
            for model_type in ModelType:
                cache_key = f"{dataset_name}_{model_type}"
                loaded_models[cache_key] = cache_key in self.model_cache
        
        loaded_models["rag_vector_store"] = self.rag_service.index is not None
        
        return loaded_models
    
    def get_rag_service(self) -> RAGService:
        """Get the RAG service instance"""
        return self.rag_service

model_manager = ModelManager() 