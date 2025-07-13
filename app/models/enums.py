from enum import Enum

class DatasetName(str, Enum):
    ANTIQUE = "antique"
    QUORA = "quora"

class ModelType(str, Enum):
    TFIDF = "tfidf"
    BERT = "bert"
    HYBRID_PARALLEL = "hybrid_parallel"
    BM25 = "bm25"

class FusionMethod(str, Enum):
    WEIGHTED = "weighted"
