from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
import string

class PunctuationRemovalHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None:
            return context
        table = str.maketrans("", "", string.punctuation)
        context.tokens = [t.translate(table) for t in context.tokens if t.translate(table)]
        return context 