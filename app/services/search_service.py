from app.models.search_models import SearchRequest, SearchResponse
from app.models.enums import ModelType
from fastapi import HTTPException

class SearchService:
    def __init__(self, strategies, result_formatter, preprocessing_service, vectorization_service, model_manager, database_service, create_ranking_service):
        self.strategies = strategies
        self.result_formatter = result_formatter
        self.preprocessing_service = preprocessing_service
        self.vectorization_service = vectorization_service
        self.model_manager = model_manager
        self.database_service = database_service
        self.create_ranking_service = create_ranking_service

    def search(self, request: SearchRequest) -> SearchResponse:
        try:
            processed_query = self.preprocessing_service.normalize(request.query)
            
            if request.model not in self.strategies:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
            
            print(f"Searching with model: {request.model}")
            results = self.strategies[request.model].search(request)
            print(f"Got {len(results)} results from strategy")
            
            doc_text_map = self.result_formatter.get_document_texts(request.dataset_name, results)
            formatted_results = self.result_formatter.format_results(results, doc_text_map)
            
            response_data = {
                "query": request.query,
                "processed_query": processed_query,
                "model": request.model,
                "results": formatted_results
            }
            
            if request.model == ModelType.HYBRID_PARALLEL and hasattr(request, 'weights'):
                response_data["weights"] = request.weights
            
            return SearchResponse(**response_data)
        except Exception as e:
            print(f"Error in SearchService.search: {e}")
            import traceback
            traceback.print_exc()
            raise 