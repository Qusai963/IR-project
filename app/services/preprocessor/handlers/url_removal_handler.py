import re
from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext

class UrlRemovalHandler(PreprocessingHandler):
    URL_PATTERN = re.compile(r"(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])")

    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        context.text = self.URL_PATTERN.sub('[URL]', context.text)
        return context 