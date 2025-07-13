from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class LemmatizationHandler(PreprocessingHandler):
    def __init__(self):
        super().__init__()
        self.lemmatizer = WordNetLemmatizer()
    def _get_wordnet_pos(self, treebank_tag: str):
        first_char = treebank_tag[0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        return tag_dict.get(first_char, wordnet.NOUN)
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None or context.pos_tags is None:
            return context
        context.tokens = [
            self.lemmatizer.lemmatize(token, pos=self._get_wordnet_pos(tag))
            for token, tag in context.pos_tags
        ]
        return context 