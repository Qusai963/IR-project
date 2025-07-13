from abc import ABC, abstractmethod
from .preprocessing_context import PreprocessingContext

class PreprocessingHandler(ABC):
    def __init__(self):
        self._next_handler = None
    def set_next(self, handler: 'PreprocessingHandler') -> 'PreprocessingHandler':
        self._next_handler = handler
        return handler
    def handle(self, context: PreprocessingContext) -> PreprocessingContext:
        context = self._process(context)
        if self._next_handler:
            return self._next_handler.handle(context)
        return context
    @abstractmethod
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        pass 