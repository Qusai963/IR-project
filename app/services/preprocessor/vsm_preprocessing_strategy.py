from .preprocessing_strategy import PreprocessingStrategy
from .handlers import (LowercaseHandler, UrlRemovalHandler, TokenizationHandler,
                       PunctuationRemovalHandler, SpellCorrectionHandler, StopWordRemovalHandler,
                       POSTaggingHandler, LemmatizationHandler, TokenJoiningHandler)
from .preprocessing_context import PreprocessingContext

class VSMPreprocessingStrategy(PreprocessingStrategy):
    def __init__(self):
        self.chain = LowercaseHandler()
        self.chain.set_next(UrlRemovalHandler()) \
                  .set_next(TokenizationHandler()) \
                  .set_next(PunctuationRemovalHandler()) \
                  .set_next(SpellCorrectionHandler()) \
                  .set_next(StopWordRemovalHandler()) \
                  .set_next(POSTaggingHandler()) \
                  .set_next(LemmatizationHandler()) \
                  .set_next(TokenJoiningHandler())

    def normalize(self, text: str) -> str:
        context = PreprocessingContext(text)
        processed_context = self.chain.handle(context)
        return processed_context.text 