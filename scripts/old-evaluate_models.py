"""
Evaluation script for IR models
Adapted from the notebook to work with the current environment structure
"""

import os
import sys
import joblib
import json
import numpy as np
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ir_datasets
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.services.ranking_service import create_ranking_service
from app.services.model_manager import model_manager
from app.models.enums import ModelType, DatasetName

class ModelEvaluator:
    def __init__(self):
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        self.model_cache = {}
        
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

    def find_similar_documents(self, query: str, dataset_name: str, model_type: ModelType, top_k: int = 10) -> List[str]:
        """Find similar documents using the specified model"""
        assets = self.load_model_assets(dataset_name, model_type)
        
        if model_type == ModelType.TFIDF:
            processed_query = self.preprocessor.normalize(query)
            vectorizer = assets["vectorizer"]
            query_vector = vectorizer.transform([processed_query])
            doc_representations = assets["doc_vectors"]
            
        elif model_type == ModelType.BERT:
            processed_query = query.strip()
            query_vector = assets["model"].encode(
                [processed_query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            doc_representations = assets["doc_embeddings"]
            
        elif model_type == ModelType.BM25:
            from app.services.bm25_service import BM25Service
            bm25_service = BM25Service()
            bm25_service.model = assets["model"]
            bm25_service.doc_ids = assets["doc_ids"]
            
            results = bm25_service.rank_documents(query, top_k)
            return [doc_id for doc_id, _ in results]
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        ranking_service = create_ranking_service(model_type, doc_representations, assets["doc_ids"])
        results = ranking_service.rank_documents(query_vector, top_k=top_k)
        
        return [doc_id for doc_id, _ in results]

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

    def get_retrieved_docs(self, dataset, dataset_name: str, model_type: ModelType, top_k: int = 10) -> Dict[str, List[str]]:
        """Get retrieved documents for all queries in the dataset"""
        retrieved_docs = {}
        
        print(f"Processing {len(list(dataset.queries_iter()))} queries...")
        for i, query in enumerate(dataset.queries_iter()):
            if i % 100 == 0:
                print(f"Processed {i} queries...")
                
            try:
                query_text = query.text
                retrieved_docs_ids = self.find_similar_documents(query_text, dataset_name, model_type, top_k)
                retrieved_docs[query.query_id] = retrieved_docs_ids
            except Exception as e:
                print(f"Error processing query {query.query_id}: {e}")
                retrieved_docs[query.query_id] = []
                
        return retrieved_docs

    def calculate_average_precision(self, ranked_list: List[str], true_relevant_docs: set) -> float:
        """Calculate average precision for a single query"""
        if not true_relevant_docs:
            return 0.0

        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(ranked_list):
            if doc_id in true_relevant_docs:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                precision_sum += precision_at_k
        
        if relevant_count == 0:
            return 0.0

        return precision_sum / relevant_count

    def calculate_map(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        average_precisions = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            query_qrels = qrel_dict[query_id]
            true_relevant_docs = {
                doc_id for doc_id, relevance in query_qrels.items() 
                if relevance > 0
            }
            
            ap = self.calculate_average_precision(retrieved, true_relevant_docs)
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

    def calculate_mean_recall(self, qrel_dict: Dict[str, Dict[str, int]], real_relevant: Dict[str, List[str]], retrieved_docs: Dict[str, List[str]]) -> float:
        """Calculate mean recall"""
        recall_scores = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            query_qrels = qrel_dict[query_id]
            relevant_docs = {
                doc_id for doc_id, relevance in query_qrels.items() 
                if relevance > 0
            }
            
            if not relevant_docs:
                continue
                
            if not retrieved:
                recall_scores.append(0.0)
                continue
                
            relevant_retrieved = [doc_id for doc_id in retrieved if doc_id in relevant_docs]
            recall = len(relevant_retrieved) / len(relevant_docs)
            recall_scores.append(recall)

        if not recall_scores:
            return 0.0
            
        return np.mean(recall_scores) * 100

    def calculate_precision_at_k(self, qrel_dict: Dict[str, Dict[str, int]], retrieved_docs: Dict[str, List[str]], k: int = 10) -> float:
        """Calculate precision at k"""
        precision_scores = []
        
        for query_id, retrieved in retrieved_docs.items():
            if query_id not in qrel_dict:
                continue
                
            retrieved_at_k = retrieved[:k]
            if not retrieved_at_k:
                precision_scores.append(0.0)
                continue
                
            relevant_retrieved = [
                doc_id for doc_id in retrieved_at_k 
                if doc_id in qrel_dict[query_id] and qrel_dict[query_id][doc_id] > 0
            ]
            precision = len(relevant_retrieved) / len(retrieved_at_k)
            precision_scores.append(precision)

        if not precision_scores:
            return 0.0
            
        return np.mean(precision_scores) * 100

    def evaluate(self, qrels_dict: Dict[str, Dict[str, int]], relevant_map: Dict[str, List[str]], retrieved_docs: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        print("Calculating evaluation metrics...")
        
        map_score = self.calculate_map(qrels_dict, retrieved_docs)
        mrr_score = self.calculate_mrr(qrels_dict, retrieved_docs)
        mean_precision = self.calculate_mean_precision(qrels_dict, retrieved_docs)
        mean_recall = self.calculate_mean_recall(qrels_dict, relevant_map, retrieved_docs)
        precision_at_10 = self.calculate_precision_at_k(qrels_dict, retrieved_docs, k=10)
        
        results = {
            "MAP": map_score,
            "MRR": mrr_score,
            "Mean_Precision": mean_precision,
            "Mean_Recall": mean_recall,
            "Precision@10": precision_at_10
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
            
        print(f"{'='*50}\n")

    def evaluate_model(self, dataset_name: str, model_type: ModelType, top_k: int = 10) -> Dict[str, float]:
        """Evaluate a specific model on a dataset"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_type.upper()} MODEL ON {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        self.check_evaluation_consistency(dataset_name, model_type)
        
        if dataset_name.lower() == "antique":
            dataset = ir_datasets.load('antique/test/non-offensive')
        elif dataset_name.lower() == "quora":
            dataset = ir_datasets.load('beir/quora/test')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        print("Extracting query relevance judgments...")
        qrels_dict, relevant_map = self.get_qrels(dataset)
        
        print(f"Retrieving documents using {model_type} model...")
        retrieved_docs = self.get_retrieved_docs(dataset, dataset_name, model_type, top_k)
        
        results = self.evaluate(qrels_dict, relevant_map, retrieved_docs)
        
        print(f"\n{'='*40} RESULTS {'='*40}")
        print(f"MAP: {results['MAP']:.2f}%")
        print(f"MRR: {results['MRR']:.2f}%")
        print(f"Mean Precision: {results['Mean_Precision']:.2f}%")
        print(f"Mean Recall: {results['Mean_Recall']:.2f}%")
        print(f"Precision@10: {results['Precision@10']:.2f}%")
        print(f"{'='*50}")
        
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
                       choices=['tfidf', 'bert', 'bm25', 'both', 'all'], 
                       help='Model type to evaluate (default: tfidf)')
    parser.add_argument('--top_k', type=int, default=10, 
                       help='Number of top documents to retrieve (default: 10)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining warning for BERT models')
    parser.add_argument('--check-models', action='store_true',
                       help='Check available models before evaluation')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    if args.check_models:
        print(f"\n{'='*50}")
        print(f"CHECKING AVAILABLE MODELS FOR {args.dataset.upper()}")
        print(f"{'='*50}")
        available_models = evaluator.check_available_models(args.dataset)
        for model_type, available in available_models.items():
            status = "✅ AVAILABLE" if available else "❌ NOT FOUND"
            print(f"{model_type.upper()}: {status}")
        print(f"{'='*50}\n")
    
    models_to_evaluate = []
    if args.model == 'both':
        models_to_evaluate = [ModelType.TFIDF, ModelType.BERT]
    elif args.model == 'all':
        models_to_evaluate = [ModelType.TFIDF, ModelType.BERT, ModelType.BM25]
    else:
        models_to_evaluate = [ModelType(args.model)]
    
    available_models = evaluator.check_available_models(args.dataset)
    models_to_skip = []
    
    for model_type in models_to_evaluate:
        if not available_models.get(model_type, False):
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
            else:
                print("\n💡 TROUBLESHOOTING TIPS:")
                print("1. Ensure models are trained: python scripts/train.py")
                print("2. Check that model files exist in saved_models directory")

if __name__ == "__main__":
    main() 