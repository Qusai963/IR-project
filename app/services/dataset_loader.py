import ir_datasets
import os
import json

from app.models.enums import DatasetName

class DatasetService:
    def __init__(self):
        self.datasets = {
            "antique": ir_datasets.load("antique/train"),
            "quora": ir_datasets.load("beir/quora/test"),
        }

    def get_queries(self, dataset_name: DatasetName, limit: int = None):
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            return []
        return list(dataset.queries_iter())[:limit]

    def get_qrels(self, dataset_name: DatasetName, limit: int = None):
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            return []
        return list(dataset.qrels_iter())[:limit]


    def get_docs(self, dataset_name: DatasetName, limit: int = None):
        dataset = self.datasets.get(dataset_name)
        if not dataset:
            return []
        return list(dataset.docs_iter())[:limit]