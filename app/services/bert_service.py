from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import torch
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

class BertService:
    def __init__(self, batch_size: int = 32):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.batch_size = batch_size
        self.preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())

    def _optimize_batch_size(self, num_docs: int) -> int:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            memory_per_doc = 768 * 4
            safe_batch_size = int((gpu_memory * 0.7) / memory_per_doc)
            return min(self.batch_size, safe_batch_size, num_docs)
        else:
            return min(self.batch_size, 32, num_docs)
  
                
    def fit_transform(self, docs: List[str]) -> Tuple[np.ndarray, SentenceTransformer]:
        num_docs = len(docs)
        batch_size = self._optimize_batch_size(num_docs)
        
        print(f"Using batch size: {batch_size} for {num_docs} documents")
        
        doc_embeddings = self.model.encode(
            docs,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=self.device
        )
        
        return doc_embeddings, self.model
      