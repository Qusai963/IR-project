from app.services.preprocessor.preprocessing_strategy import PreprocessingStrategy

class PreprocessingService:
    def __init__(self, strategy: PreprocessingStrategy):
        self.strategy = strategy

    def normalize(self, text: str) -> str:
        return self.strategy.normalize(text) 