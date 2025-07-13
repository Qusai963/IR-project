from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np
from app.services.preprocessor.handlers.tokenization_handler import TokenizationHandler
from app.services.preprocessor.preprocessing_context import PreprocessingContext
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

class BM25Service:
    def __init__(self):
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        self.model = None
        self.doc_ids = []
        
    def _tokenize(self, text: str) -> List[str]:
        processed_text = self.preprocessor.normalize(text)
        tokenizer = TokenizationHandler()
        context = PreprocessingContext(processed_text)
        processed_context = tokenizer.handle(context)
        return processed_context.tokens
    
    def fit_transform(self, processed_texts: List[str], doc_ids: List[str]) -> Tuple[BM25Okapi, List[str]]:
        self.doc_ids = doc_ids
        
        tokenized_docs = [self._tokenize(doc) for doc in processed_texts]
        
        self.model = BM25Okapi(tokenized_docs)
        return self.model, doc_ids
    
    def get_scores(self, query: str) -> List[float]:
        if self.model is None:
            raise ValueError("BM25 model not fitted. Call fit_transform first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.model.get_scores(tokenized_query)
        return scores
    
    def rank_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        scores = self.get_scores(query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))
        
        return results 