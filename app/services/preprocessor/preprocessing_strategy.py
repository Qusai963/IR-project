from abc import ABC, abstractmethod

class PreprocessingStrategy(ABC):
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass 