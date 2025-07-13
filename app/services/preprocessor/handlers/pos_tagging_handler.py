from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
from nltk import pos_tag

class POSTaggingHandler(PreprocessingHandler):
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None:
            return context
        context.pos_tags = pos_tag(context.tokens)
        return context 