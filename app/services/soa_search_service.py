import requests
from typing import Dict, Any, List
from app.models.search_models import SearchRequest, SearchResponse, SearchResult
from app.models.enums import ModelType
from fastapi import HTTPException

class SOASearchService:
    """
    SOA-compliant search service that communicates with other services via HTTP APIs
    instead of direct method calls.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform search using SOA architecture:
        1. Call preprocessing service to normalize query
        2. Call vectorization service to vectorize query
        3. Call ranking service to rank documents
        4. Call database service to get document texts
        """
        try:
            preprocess_response = self._call_preprocessing_service(request.query, request.model)
            processed_query = preprocess_response["processed"]
            
            vectorize_response = self._call_vectorization_service(
                processed_query, request.model, request.dataset_name
            )
            
            rank_response = self._call_ranking_service(
                vectorize_response["query_vector"],
                request.model,
                request.dataset_name,
                request.top_k
            )
            
            print(f"Ranking response: {rank_response}")
            print(f"Results type: {type(rank_response.get('results', []))}")
            if rank_response.get('results'):
                print(f"First result: {rank_response['results'][0]}")
            
            doc_ids = []
            results = rank_response.get("results", [])
            
            if not results:
                print("No results returned from ranking service")
                return SearchResponse(
                    query=request.query,
                    processed_query=processed_query,
                    model=request.model,
                    results=[]
                )
            
            for result in results:
                if isinstance(result, dict) and "doc_id" in result:
                    doc_ids.append(result["doc_id"])
                else:
                    print(f"Unexpected result format: {result}")
            
            print(f"Extracted doc_ids: {doc_ids}")
            
            if not doc_ids:
                print("No valid doc_ids extracted")
                return SearchResponse(
                    query=request.query,
                    processed_query=processed_query,
                    model=request.model,
                    results=[]
                )
            
            doc_texts = self._call_database_service(
                request.dataset_name,
                doc_ids
            )
            
            formatted_results = []
            print(f"Formatting {len(rank_response['results'])} results")
            print(f"Doc texts received: {len(doc_texts)} documents")
            
            for i, rank_result in enumerate(rank_response["results"]):
                print(f"Processing result {i}: {rank_result}")
                try:
                    doc_id = rank_result["doc_id"]
                    score = rank_result["score"]
                    doc_text = next((doc["text"] for doc in doc_texts if doc.get("doc_id") == doc_id or doc.get("id") == doc_id), "")
                    
                    formatted_results.append(SearchResult(
                        doc_id=doc_id,
                        score=score,
                        text=doc_text
                    ))
                    print(f"Successfully formatted result {i}")
                except Exception as e:
                    print(f"Error formatting result {i}: {e}")
                    print(f"Result data: {rank_result}")
                    raise
            
            return SearchResponse(
                query=request.query,
                processed_query=processed_query,
                model=request.model,
                results=formatted_results
            )
            
        except requests.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Service communication failed: {str(e)}")
        except Exception as e:
            import traceback
            print(f"Error in SOA search: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def _call_preprocessing_service(self, text: str, model_type: ModelType) -> Dict[str, Any]:
        """Call preprocessing service via API"""
        strategy = "embedding" if model_type == ModelType.BERT else "vsm"
        response = requests.post(
            f"{self.base_url}/api/preprocess",
            json={"text": text, "strategy": strategy}
        )
        response.raise_for_status()
        return response.json()
    
    def _call_vectorization_service(self, query: str, model_type: ModelType, dataset_name) -> Dict[str, Any]:
        """Call vectorization service via API"""
        response = requests.post(
            f"{self.base_url}/api/vectorize",
            json={
                "query": query,
                "model_type": model_type.value,
                "dataset_name": dataset_name.value
            }
        )
        response.raise_for_status()
        return response.json()
    
    def _call_ranking_service(self, query_vector: List[float], model_type: ModelType, dataset_name, top_k: int) -> Dict[str, Any]:
        """Call ranking service via API"""
        response = requests.post(
            f"{self.base_url}/api/rank",
            json={
                "query_vector": query_vector,
                "model_type": model_type.value,
                "dataset_name": dataset_name.value,
                "top_k": top_k
            }
        )
        response.raise_for_status()
        return response.json()
    
    def _call_database_service(self, dataset_name, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Call database service via API to get document texts"""
        response = requests.post(
            f"{self.base_url}/api/database/documents",
            json={
                "dataset_name": dataset_name.value,
                "doc_ids": doc_ids
            }
        )
        response.raise_for_status()
        return response.json()["documents"]
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all dependent services"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return {"status": "unhealthy", "error": "Cannot reach services"} 