from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext

class TokenJoiningHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None:
            return context
        context.text = " ".join(context.tokens)
        return context 