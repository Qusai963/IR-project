from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from app.services.preprocessor.vsm_preprocessing_strategy import VSMPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService

class QueryRefinementService:
    def __init__(self):
        self.preprocessor = PreprocessingService(VSMPreprocessingStrategy())
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.query_history = []
        self.synonym_cache = {}

    def get_synonyms(self, word: str) -> List[str]:
        if word in self.synonym_cache:
            return self.synonym_cache[word]

        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.add(lemma.name().replace('_', ' '))
        
        self.synonym_cache[word] = list(synonyms)
        return list(synonyms)

    def expand_query(self, query: str) -> List[str]:
        tokens = word_tokenize(query.lower())
        expanded_queries = [query]

        for token in tokens:
            if len(token) > 3:
                synonyms = self.get_synonyms(token)
                for synonym in synonyms[:2]:
                    new_query = query.replace(token, synonym)
                    expanded_queries.append(new_query)

        return expanded_queries

    def suggest_improvements(self, query: str) -> Dict[str, List[str]]:
        processed_query = self.preprocessor.normalize(query)
        expanded_queries = self.expand_query(processed_query)
        self.query_history.append(processed_query)
        if len(self.query_history) > 100:
            self.query_history.pop(0)
        return {
            "expanded_queries": expanded_queries,
            "processed_query": processed_query
        } 