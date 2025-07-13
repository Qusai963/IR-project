from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple
from scipy.sparse import csr_matrix
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

from app.services.preprocessor.handlers.tokenization_handler import TokenizationHandler
from app.services.preprocessor.preprocessing_context import PreprocessingContext

class TfidfService:
    def __init__(self):
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        self.vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess,
            tokenizer=self._tokenize,
            lowercase=False,  
            token_pattern=None
        )

    def _preprocess(self, text: str) -> str:
        return self.preprocessor.normalize(text)

    def _tokenize(self, text: str) -> List[str]:
        tokenizer = TokenizationHandler()
        context = PreprocessingContext(text)
        processed_context = tokenizer.handle(context)
        return processed_context.tokens

    def fit_transform(self, raw_texts: List[str]) -> Tuple[csr_matrix, TfidfVectorizer]:
        doc_vectors = self.vectorizer.fit_transform(raw_texts)
        return doc_vectors, self.vectorizer
