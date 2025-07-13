from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
import unicodedata

class UnicodeNormalizationHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        context.text = unicodedata.normalize("NFKD", context.text)
        return context 