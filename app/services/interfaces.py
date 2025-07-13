from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from app.models.enums import DatasetName, ModelType

class IPreprocessingService(ABC):
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass

class IDatabaseService(ABC):
    
    @abstractmethod
    def get_documents_by_ids(self, dataset_name: DatasetName, doc_ids: List[str]) -> List[Dict[str, Any]]:
        pass

class IModelManager(ABC):
    
    @abstractmethod
    def get_model_assets(self, dataset_name: DatasetName, model_type: ModelType) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def is_model_loaded(self, dataset_name: DatasetName, model_type: ModelType) -> bool:
        pass

class IVectorizationService(ABC):
    
    @abstractmethod
    def vectorize_query(self, query: str, model_type: ModelType, assets: Dict[str, Any]) -> Tuple[Any, Any]:
        pass

class IRankingService(ABC):
    
    @abstractmethod
    def rank_documents(self, query_vector: Any, doc_representations: Any, doc_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        pass 