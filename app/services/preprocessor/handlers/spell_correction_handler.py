from ..preprocessing_handler import PreprocessingHandler
from ..preprocessing_context import PreprocessingContext
from spellchecker import SpellChecker

class SpellCorrectionHandler(PreprocessingHandler):
    def __init__(self):
        super().__init__()
        self.spell_checker = SpellChecker(distance=1)
    def _process(self, context: PreprocessingContext) -> PreprocessingContext:
        if context.tokens is None:
            return context
        misspelled = self.spell_checker.unknown(context.tokens)
        corrected_tokens = []
        for token in context.tokens:
            if token in misspelled:
                correction = self.spell_checker.correction(token)
                corrected_tokens.append(correction if correction else token)
            else:
                corrected_tokens.append(token)
        context.tokens = corrected_tokens
        return context 