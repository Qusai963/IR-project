from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
from nltk.corpus import stopwords

class StopWordRemovalHandler(PreprocessingHandler):
    def __init__(self):
        super().__init__()
        self.stop_words = set(stopwords.words("english"))
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None:
            return context
        context.tokens = [t for t in context.tokens if t not in self.stop_words]
        return context 