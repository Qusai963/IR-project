from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext

class LowercaseHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        context.text = context.text.lower()
        return context 