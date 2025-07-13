import nltk
from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext

class TokenizationHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        context.tokens = nltk.word_tokenize(context.text)
        return context 