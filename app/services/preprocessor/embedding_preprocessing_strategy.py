from .preprocessing_strategy import PreprocessingStrategy
from .handlers import LowercaseHandler, UrlRemovalHandler
from .preprocessing_context import PreprocessingContext

class EmbeddingPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self):
        self.chain = LowercaseHandler()
        self.chain.set_next(UrlRemovalHandler())

    def normalize(self, text: str) -> str:
        context = PreprocessingContext(text)
        processed_context = self.chain.handle(context)
        return processed_context.text 