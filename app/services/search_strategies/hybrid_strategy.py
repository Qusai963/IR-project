from .interfaces import ISearchStrategy
from app.models.search_models import SearchRequest
from .tfidf_strategy import TFIDFSearchStrategy
from .bert_strategy import BERTSearchStrategy
from .bm25_strategy import BM25SearchStrategy
from app.models.enums import ModelType
from fastapi import HTTPException
from typing import List, Dict, Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class HybridSearchStrategy(ISearchStrategy):
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.tfidf_strategy = TFIDFSearchStrategy(model_manager)
        self.bert_strategy = BERTSearchStrategy(model_manager)
        self.bm25_strategy = BM25SearchStrategy(model_manager)
        self.scaler = MinMaxScaler()

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        scores_array = np.array(scores).reshape(-1, 1)
        normalized = self.scaler.fit_transform(scores_array)
        return normalized.flatten().tolist()

    def _get_model_scores(self, strategy, model_type: ModelType, request: SearchRequest, all_doc_ids: List[str]) -> Dict[str, float]:
        """Get scores from a specific model strategy"""
        try:
            temp_request = SearchRequest(
                dataset_name=request.dataset_name,
                query=request.query,
                model=model_type,
                top_k=len(all_doc_ids),
                weights=request.weights
            )
            
            results = strategy.search(temp_request)
            
            scores_dict = {}
            for item in results:
                if isinstance(item, tuple) and len(item) == 2:
                    doc_id, score = item
                    scores_dict[doc_id] = score
                elif isinstance(item, dict) and 'doc_id' in item and 'score' in item:
                    scores_dict[item['doc_id']] = item['score']
                else:
                    print(f"WARNING: Unexpected result format for {model_type}: {item}")
                    continue
            
            for doc_id in all_doc_ids:
                if doc_id not in scores_dict:
                    scores_dict[doc_id] = 0.0
                    
            return scores_dict
        except Exception as e:
            print(f"Error in _get_model_scores for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            return {doc_id: 0.0 for doc_id in all_doc_ids}

    def search(self, request: SearchRequest) -> List[tuple]:
        """Perform hybrid search by reusing individual strategies"""
        try:
            tfidf_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.TFIDF)
            if not tfidf_assets:
                self.model_manager._load_model_assets(request.dataset_name, ModelType.TFIDF)
                tfidf_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.TFIDF)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load TF-IDF model: {e}"
            )

        try:
            bert_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.BERT)
            if not bert_assets:
                self.model_manager._load_model_assets(request.dataset_name, ModelType.BERT)
                bert_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.BERT)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load BERT model: {e}"
            )

        bm25_assets = None
        try:
            bm25_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.BM25)
            if not bm25_assets:
                self.model_manager._load_model_assets(request.dataset_name, ModelType.BM25)
                bm25_assets = self.model_manager.get_model_assets(request.dataset_name, ModelType.BM25)
        except Exception as e:
            print(f"BM25 model not available: {e}")
            bm25_assets = None

        all_doc_ids = set(tfidf_assets["doc_ids"])
        all_doc_ids.update(bert_assets["doc_ids"])
        if bm25_assets:
            all_doc_ids.update(bm25_assets["doc_ids"])
        
        all_doc_ids = list(all_doc_ids)
        
        tfidf_scores = self._get_model_scores(self.tfidf_strategy, ModelType.TFIDF, request, all_doc_ids)
        bert_scores = self._get_model_scores(self.bert_strategy, ModelType.BERT, request, all_doc_ids)
        
        tfidf_score_list = [tfidf_scores[doc_id] for doc_id in all_doc_ids]
        bert_score_list = [bert_scores[doc_id] for doc_id in all_doc_ids]
        
        bm25_scores = None
        if bm25_assets:
            bm25_scores = self._get_model_scores(self.bm25_strategy, ModelType.BM25, request, all_doc_ids)
        bm25_score_list = [bm25_scores[doc_id] if bm25_scores else 0.0 for doc_id in all_doc_ids]
        
        models_to_use = []
        weights_to_use = []
        
        if request.weights:
            provided_weights = list(request.weights)
        else:
            provided_weights = [1/3, 1/3, 1/3]
        
        if len(provided_weights) > 0 and provided_weights[0] > 0:
            models_to_use.append(('tfidf', tfidf_score_list))
            weights_to_use.append(provided_weights[0])
        
        if len(provided_weights) > 1 and provided_weights[1] > 0:
            models_to_use.append(('bert', bert_score_list))
            weights_to_use.append(provided_weights[1])
        
        if len(provided_weights) > 2 and provided_weights[2] > 0:
            models_to_use.append(('bm25', bm25_score_list))
            weights_to_use.append(provided_weights[2])
        
        if weights_to_use:
            total_weight = sum(weights_to_use)
            weights_to_use = [w / total_weight for w in weights_to_use]
        
        if len(models_to_use) > 1:
            normalized_scores = []
            for model_name, score_list in models_to_use:
                normalized_scores.append(self._normalize_scores(score_list))
            
            combined_scores = []
            for i, doc_id in enumerate(all_doc_ids):
                combined_score = 0.0
                for j, (model_name, _) in enumerate(models_to_use):
                    combined_score += weights_to_use[j] * normalized_scores[j][i]
                combined_scores.append((doc_id, combined_score))
        else:
            score_list = models_to_use[0][1]
            combined_scores = [(doc_id, score) for doc_id, score in zip(all_doc_ids, score_list)]

        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:request.top_k] 