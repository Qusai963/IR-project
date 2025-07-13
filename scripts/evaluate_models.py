"""
Evaluation script for IR models
Adapted from the notebook to work with the current environment structure
"""

import os
import sys
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ir_datasets
from app.services.model_manager import model_manager
from app.models.enums import ModelType, DatasetName
from app.services.search_strategies.bm25_strategy import BM25SearchStrategy
from app.services.search_strategies.tfidf_strategy import TFIDFSearchStrategy
from app.services.search_strategies.hybrid_strategy import HybridSearchStrategy
from app.services.search_strategies.bert_strategy import BERTSearchStrategy
from app.models.search_models import SearchRequest

class ModelEvaluator:
    def __init__(self):
        self.model_cache = {}
        
        self.strategies = {
            ModelType.BM25: BM25SearchStrategy(model_manager),
            ModelType.TFIDF: TFIDFSearchStrategy(model_manager),
            ModelType.BERT: BERTSearchStrategy(model_manager),
            ModelType.HYBRID_PARALLEL: HybridSearchStrategy(model_manager),
        }
        
    def load_model_assets(self, dataset_name: str, model_type: ModelType) -> Dict[str, Any]:
        """Load model assets using the centralized ModelManager"""
        cache_key = f"{dataset_name}_{model_type}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        print(f"Loading model assets for dataset: {dataset_name}, model: {model_type}...")
        
        try:
            dataset_enum = DatasetName(dataset_name.lower())
        except ValueError:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        assets = model_manager.get_model_assets(dataset_enum, model_type)
        if assets is None:
            try:
                model_manager._load_model_assets(dataset_enum, model_type)
                assets = model_manager.get_model_assets(dataset_enum, model_type)
                if assets is None:
                    raise FileNotFoundError(f"Failed to load model for '{dataset_name}'")
            except Exception as e:
                raise FileNotFoundError(f"Model for '{dataset_name}' not found. Please run the training script first. Error: {e}")
        
        self.model_cache[cache_key] = assets
        print("Assets loaded successfully.")
        return assets

    def find_similar_documents(self, query: str, dataset_name: str, model_type: ModelType, top_k: int = 10, weights: tuple = (0.33, 0.33, 0.34)) -> List[str]:
        """Find similar documents using the specified model, leveraging existing search strategies"""
        try:
            dataset_enum = DatasetName(dataset_name.lower())
            
            if model_type == ModelType.HYBRID_PARALLEL:
                self.load_model_assets(dataset_name, ModelType.TFIDF)
                self.load_model_assets(dataset_name, ModelType.BERT)
                try:
                    self.load_model_assets(dataset_name, ModelType.BM25)
                except Exception:
                    pass
            else:
                self.load_model_assets(dataset_name, model_type)
            
            if model_type == ModelType.HYBRID_PARALLEL:
                if len(weights) == 2:
                    weights_list = list(weights) + [0.0]
                elif len(weights) == 3:
                    weights_list = list(weights)
                else:
                    weights_list = [0.33, 0.33, 0.34]
            else:
                weights_list = [0.5, 0.5]
            
            search_request = SearchRequest(
                dataset_name=dataset_enum,
                query=query,
                model=model_type,
                top_k=top_k,
                weights=weights_list
            )
            
            if model_type in self.strategies:
                results = self.strategies[model_type].search(search_request)
                return [doc_id for doc_id, _ in results]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            print(f"Error in find_similar_documents for {model_type}: {e}")
            return []

    def get_qrels(self, dataset) -> Tuple[Dict[str, Dict[str, int]], Dict[str, List[str]]]:
        """Extract query relevance judgments from dataset"""
        qrel_dict = defaultdict(dict)
        real_relevant = defaultdict(list)
        
        for qrel in dataset.qrels_iter():
            query_id = qrel[0]
            doc_id = qrel[1]
            relevance = qrel[2]
            qrel_dict[query_id][doc_id] = relevance
            if relevance > 0:
                real_relevant[query_id].append(doc_id)
                
        return dict(qrel_dict), dict(real_relevant)

    def get_retrieved_docs(self, dataset, dataset_name: str, model_type: ModelType, top_k: int = 10, weights: tuple = (0.33, 0.33, 0.34)) -> Dict[str, List[str]]:
        """Get retrieved documents for all queries in the dataset, including hybrid"""
        retrieved_docs = {}
        for i, query in enumerate(dataset.queries_iter()):
            if i > 0 and i % 2000 == 0:
                print(f"Processed {i} queries...")
            try:
                retrieved_docs_ids = self.find_similar_documents(
                    query.text, dataset_name, model_type, top_k, weights
                )
                retrieved_docs[query.query_id] = retrieved_docs_ids
            except Exception as e:
                retrieved_docs[query.query_id] = []
        return retrieved_docs
    def calculate_average_precision(self, retrieved: List[str], query_docs: set) -> float:
        """Calculate average precision for a single query"""
        if not query_docs:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in query_docs:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                precision_sum += precision_at_k
        
        if relevant_count == 0:
            return 0.0

        return precision_sum / relevant_count

    def calculate_precision_at_k(self, retrieved: List[str], query_docs: set, top_k: int): 
        if not query_docs:
            return 0.0

        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if i >= top_k:
                break
            if doc_id in query_docs:
                relevant_count += 1
        
        if relevant_count == 0:
            return 0.0

        return relevant_count / min(len(retrieved), top_k)
  
    def calculate_map(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        average_precisions = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            query_qrels = qrel_dict[query_id]
            query_docs = {
                doc_id for doc_id, relevance in query_qrels.items() 
                if relevance > 0
            }
            
            ap = self.calculate_average_precision(retrieved, query_docs)
            average_precisions.append(ap)
        
        if not average_precisions:
            return 0.0
       
        return np.mean(average_precisions) * 100

    def calculate_mrr(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        reciprocal_ranks = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            found_relevant = False
            for i, doc_id in enumerate(retrieved):
                if doc_id in qrel_dict[query_id] and qrel_dict[query_id][doc_id] > 0:
                    reciprocal_ranks.append(1 / (i + 1))
                    found_relevant = True
                    break
            
            if not found_relevant:
                reciprocal_ranks.append(0.0)
                    
        if not reciprocal_ranks:
            return 0.0
            
        return np.mean(reciprocal_ranks) * 100

    def calculate_mean_precision(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]]) -> float:
        """Calculate mean precision"""
        precision_scores = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            if not retrieved:
                precision_scores.append(0.0)
                continue
                
            relevant_retrieved = [
                doc_id for doc_id in retrieved 
                if doc_id in qrel_dict[query_id] and qrel_dict[query_id][doc_id] > 0
            ]
            precision = len(relevant_retrieved) / len(retrieved)
            precision_scores.append(precision)

        if not precision_scores:
            return 0.0
            
        return np.mean(precision_scores) * 100

    def evaluate(self, qrels_dict: Dict[str, Dict[str, int]], relevant_map: Dict[str, List[str]], retrieved_docs: Dict[str, List[str]], top_k: int) -> Dict[str, float]:
        """Calculate only MAP and MRR evaluation metrics"""
        map_score = self.calculate_map(qrels_dict, retrieved_docs)
        mrr_score = self.calculate_mrr(qrels_dict, retrieved_docs)
        results = {
            "MAP": map_score,
            "MRR": mrr_score
        }
        return results

    def check_evaluation_consistency(self, dataset_name: str, model_type: ModelType) -> None:
        """Check for potential evaluation consistency issues"""
        print(f"\n{'='*50}")
        print(f"EVALUATION CONSISTENCY CHECK FOR {model_type.upper()}")
        print(f"{'='*50}")
        
        if model_type == ModelType.BERT:
            print("⚠️  IMPORTANT: BERT Evaluation Checks:")
            print("1. Ensure BERT models were trained with the latest fixes")
            print("2. Check that document embeddings use normalize_embeddings=True")
            print("3. Verify query preprocessing matches training (minimal preprocessing)")
            print("4. If using old models, retrain with updated bert_service.py")
            
            try:
                assets = self.load_model_assets(dataset_name, model_type)
                print("✅ Model assets loaded successfully")
                
                if assets["doc_embeddings"] is not None:
                    print(f"✅ Document embeddings shape: {assets['doc_embeddings'].shape}")
                    
                    sample_embeddings = assets["doc_embeddings"][:100]
                    norms = np.linalg.norm(sample_embeddings, axis=1)
                    avg_norm = np.mean(norms)
                    print(f"✅ Average L2 norm of sample embeddings: {avg_norm:.4f}")
                    
                    if abs(avg_norm - 1.0) > 0.1:
                        print("⚠️  WARNING: Embeddings may not be normalized!")
                        print("   This could indicate using old models. Consider retraining.")
                    else:
                        print("✅ Embeddings appear to be properly normalized")
                        
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("   Please ensure models are trained before evaluation")
        
        elif model_type == ModelType.TFIDF:
            print("✅ TF-IDF evaluation should be consistent with training")
            
        elif model_type == ModelType.BM25:
            print("✅ BM25 evaluation should be consistent with training")
            try:
                assets = self.load_model_assets(dataset_name, model_type)
                print("✅ BM25 model loaded successfully")
                if assets["model"] is not None:
                    print("✅ BM25 model instance available")
            except Exception as e:
                print(f"❌ Error loading BM25 model: {e}")
                print("   Please ensure BM25 models are trained before evaluation")
        
        elif model_type == ModelType.HYBRID_PARALLEL:
            print("⚠️  HYBRID PARALLEL Evaluation Checks:")
            print("1. Ensure all individual models (TF-IDF, BERT, BM25) are trained")
            print("2. Check that hybrid strategy uses the same preprocessing as individual models")
            print("3. Verify weight normalization and model exclusion logic")
            print("4. Hybrid uses the same search strategies as individual models")
            
            try:
                tfidf_assets = self.load_model_assets(dataset_name, ModelType.TFIDF)
                print("✅ TF-IDF model loaded successfully")
                
                bert_assets = self.load_model_assets(dataset_name, ModelType.BERT)
                print("✅ BERT model loaded successfully")
                
                try:
                    bm25_assets = self.load_model_assets(dataset_name, ModelType.BM25)
                    print("✅ BM25 model loaded successfully")
                except Exception:
                    print("⚠️  BM25 model not available - hybrid will use TF-IDF + BERT only")
                    
                print("✅ All required models for hybrid search are available")
                
            except Exception as e:
                print(f"❌ Error loading models for hybrid search: {e}")
                print("   Please ensure TF-IDF and BERT models are trained before hybrid evaluation")
            
        print(f"{'='*50}\n")

    def evaluate_model(self, dataset_name: str, model_type: ModelType, top_k: int = 10, weights: tuple = (0.33, 0.33, 0.34)) -> Dict[str, float]:
        """Evaluate a specific model on a dataset, including hybrid"""
        self.check_evaluation_consistency(dataset_name, model_type)
        if dataset_name.lower() == "antique":
            dataset = ir_datasets.load('antique/test/non-offensive')
        elif dataset_name.lower() == "quora":
            dataset = ir_datasets.load('beir/quora/test')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        qrels_dict, relevant_map = self.get_qrels(dataset)
        retrieved_docs = self.get_retrieved_docs(dataset, dataset_name, model_type, top_k, weights)
        results = self.evaluate(qrels_dict, relevant_map, retrieved_docs, top_k)
        print(f"MAP: {results['MAP']:.2f}%  |  MRR: {results['MRR']:.2f}%")
        return results

    def check_available_models(self, dataset_name: str) -> Dict[str, bool]:
        """Check which models are available for a dataset"""
        available_models = {}
        
        try:
            dataset_enum = DatasetName(dataset_name.lower())
        except ValueError:
            print(f"⚠️  Unsupported dataset: {dataset_name}")
            return available_models
        
        for model_type in ModelType:
            try:
                if model_type == ModelType.TFIDF:
                    model_path = os.path.join(model_manager.saved_models_dir, dataset_enum, "tfidf")
                elif model_type == ModelType.BM25:
                    model_path = os.path.join(model_manager.saved_models_dir, dataset_enum, "bm25")
                else:
                    model_path = os.path.join(model_manager.saved_models_dir, dataset_enum, "bert")
                
                available_models[model_type] = os.path.exists(model_path)
            except Exception:
                available_models[model_type] = False
        
        return available_models

def main():
    parser = argparse.ArgumentParser(description='Evaluate IR models on datasets')
    parser.add_argument('--dataset', type=str, default='antique', 
                       choices=['antique', 'quora'], 
                       help='Dataset to evaluate on (default: antique)')
    parser.add_argument('--model', type=str, default='tfidf', 
                       choices=['tfidf', 'bert', 'bm25', 'hybrid_parallel', 'both', 'all'], 
                       help='Model type to evaluate (default: tfidf)')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Number of top documents to retrieve (default: 10)')
    parser.add_argument('--weights', type=float, nargs='+', default=[0.33, 0.33, 0.34], 
                       help='Weights for hybrid fusion [tfidf, bert, bm25] (2 or 3 values, default: 0.33 0.33 0.34)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining warning for BERT models')
    parser.add_argument('--check-models', action='store_true',
                       help='Check available models before evaluation')
    args = parser.parse_args()
    evaluator = ModelEvaluator()
    if args.check_models:
        available_models = evaluator.check_available_models(args.dataset)
        for model_type, available in available_models.items():
            status = "✅ AVAILABLE" if available else "❌ NOT FOUND"
            print(f"{model_type.upper()}: {status}")
    models_to_evaluate = []
    if args.model == 'both':
        models_to_evaluate = [ModelType.TFIDF, ModelType.BERT]
    elif args.model == 'all':
        models_to_evaluate = [ModelType.TFIDF, ModelType.BERT, ModelType.BM25, ModelType.HYBRID_PARALLEL]
    elif args.model == 'hybrid_parallel':
        models_to_evaluate = [ModelType.HYBRID_PARALLEL]
    else:
        models_to_evaluate = [ModelType(args.model)]
    available_models = evaluator.check_available_models(args.dataset)
    models_to_skip = []
    for model_type in models_to_evaluate:
        if model_type == ModelType.HYBRID_PARALLEL:
            if not (available_models.get(ModelType.TFIDF, False) and available_models.get(ModelType.BERT, False)):
                print(f"⚠️  {model_type.upper()} requires both TF-IDF and BERT models for {args.dataset}. Skipping...")
                models_to_skip.append(model_type)
        elif not available_models.get(model_type, False):
            print(f"⚠️  {model_type.upper()} model not found for {args.dataset}. Skipping...")
            models_to_skip.append(model_type)
    models_to_evaluate = [m for m in models_to_evaluate if m not in models_to_skip]
    if not models_to_evaluate:
        print(f"❌ No models available for evaluation on {args.dataset} dataset.")
        print("💡 Please train models first:")
        print("   - TF-IDF/BERT: python scripts/train.py --dataset quora")
        print("   - BM25: python scripts/train_bm25.py --dataset quora")
        return
    for model_type in models_to_evaluate:
        try:
            if model_type == ModelType.HYBRID_PARALLEL:
                evaluator.evaluate_model(args.dataset, model_type, args.top_k, weights=tuple(args.weights))
            else:
                evaluator.evaluate_model(args.dataset, model_type, args.top_k)
        except Exception as e:
            print(f"Error evaluating {model_type} model: {e}")
            if model_type == ModelType.BERT:
                print("\n💡 TROUBLESHOOTING TIPS:")
                print("1. Ensure BERT models are trained: python scripts/train.py --model bert")
                print("2. Check that models use the latest bert_service.py")
                print("3. Verify embedding normalization consistency")
                print("4. Use --force-retrain flag to get retraining warnings")
            elif model_type == ModelType.BM25:
                print("\n💡 TROUBLESHOOTING TIPS:")
                print("1. Ensure BM25 models are trained: python scripts/train_bm25.py")
                print("2. Check that BM25 model files exist in saved_models directory")
            elif model_type == ModelType.HYBRID_PARALLEL:
                print("\n💡 TROUBLESHOOTING TIPS:")
                print("1. Ensure TF-IDF and BERT models are trained: python scripts/train.py")
                print("2. BM25 is optional but recommended: python scripts/train_bm25.py")
                print("3. Check that hybrid strategy is properly configured")
                print("4. Verify weight format: [tfidf_weight, bert_weight, bm25_weight]")
                print("5. Use --weights 0.5 0.5 0 to exclude BM25")
            else:
                print("\n💡 TROUBLESHOOTING TIPS:")
                print("1. Ensure models are trained: python scripts/train.py")
                print("2. Check that model files exist in saved_models directory")

if __name__ == "__main__":
    main() 