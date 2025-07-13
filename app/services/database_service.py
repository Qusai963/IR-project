import psycopg2
from typing import List, Dict, Any, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

from app.models.enums import DatasetName
from app.services.interfaces import IDatabaseService

class DatabaseService(IDatabaseService):
    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                dbname=os.environ.get("dbname"),
                user=os.environ.get("user"),
                password=os.environ.get("password"),
                host=os.environ.get("host"),
                port=os.environ.get("port")
            )
            self.cur = self.conn.cursor()
            print("Successfully connected to PostgreSQL.")
        except psycopg2.OperationalError as e:
            print(f"Could not connect to PostgreSQL database. Please ensure it is running and credentials are correct. Error: {e}")
            raise

    def create_documents_table(self) -> None:
        create_table_command = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL,
            text TEXT NOT NULL,
            vsm_text TEXT,
            embedding_text TEXT,
            dataset_name VARCHAR(100) NOT NULL,
            UNIQUE(dataset_name, doc_id)
        );
        """
        self.cur.execute(create_table_command)
        self.conn.commit()
        print("'documents' table created or already exists.")

    def insert_docs(self, docs: List[Dict[str, Any]], dataset_name: DatasetName) -> None:
        has_vsm = any('vsm_text' in doc for doc in docs)
        has_embedding = any('embedding_text' in doc for doc in docs)
        if has_vsm or has_embedding:
            query = "INSERT INTO documents (doc_id, text, vsm_text, embedding_text, dataset_name) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (dataset_name, doc_id) DO NOTHING;"
            data_to_insert = [(
                doc['doc_id'],
                doc['text'],
                doc.get('vsm_text'),
                doc.get('embedding_text'),
                dataset_name
            ) for doc in docs]
        else:
            query = "INSERT INTO documents (doc_id, text, dataset_name) VALUES (%s, %s, %s) ON CONFLICT (dataset_name, doc_id) DO NOTHING;"
            data_to_insert = [(doc['doc_id'], doc['text'], dataset_name) for doc in docs]
        self.cur.executemany(query, data_to_insert)
        self.conn.commit()
        print(f"Inserted {len(data_to_insert)} documents into the database for dataset '{dataset_name}'.")

    def get_docs_by_dataset(self, dataset_name: DatasetName, limit: int = None) -> List[Dict[str, Any]]:
        query = "SELECT doc_id, text, vsm_text, embedding_text FROM documents WHERE dataset_name = %s"
        params = [dataset_name]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        self.cur.execute(query, tuple(params))
        docs = [{
            "doc_id": row[0],
            "text": row[1],
            "vsm_text": row[2],
            "embedding_text": row[3]
        } for row in self.cur.fetchall()]
        return docs

    def get_processed_docs_by_dataset(self, dataset_name: DatasetName, limit: int = None) -> Tuple[List[str], List[str], List[str], List[str]]:
        query = "SELECT doc_id, text, vsm_text, embedding_text FROM documents WHERE dataset_name = %s"
        params = [dataset_name]
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        self.cur.execute(query, tuple(params))
        doc_ids = []
        raw_texts = []
        vsm_texts = []
        embedding_texts = []
        for row in self.cur.fetchall():
            doc_ids.append(row[0])
            raw_texts.append(row[1])
            vsm_texts.append(row[2])
            embedding_texts.append(row[3])
        return doc_ids, raw_texts, vsm_texts, embedding_texts

    def __del__(self):
        if hasattr(self, 'cur') and self.cur:
            self.cur.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        print("PostgreSQL connection closed.")

    def get_documents_by_ids(self, dataset_name: DatasetName, doc_ids: list[str]) -> list[dict]:
        if not doc_ids:
            return []
        placeholders = ','.join(['%s'] * len(doc_ids))
        query = f"""
            SELECT doc_id, text, vsm_text, embedding_text 
            FROM documents 
            WHERE dataset_name = %s AND doc_id IN ({placeholders})
        """
        self.cur.execute(query, [dataset_name] + doc_ids)
        rows = self.cur.fetchall()
        doc_map = {row[0]: {'id': row[0], 'text': row[1], 'vsm_text': row[2], 'embedding_text': row[3]} for row in rows}
        ordered_docs = [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]
        return ordered_docs

    def update_vsm_text(self, doc_id: str, dataset_name: DatasetName, vsm_text: str) -> None:
        query = "UPDATE documents SET vsm_text = %s WHERE doc_id = %s AND dataset_name = %s;"
        self.cur.execute(query, (vsm_text, doc_id, dataset_name))
        self.conn.commit()

    def update_embedding_text(self, doc_id: str, dataset_name: DatasetName, embedding_text: str) -> None:
        query = "UPDATE documents SET embedding_text = %s WHERE doc_id = %s AND dataset_name = %s;"
        self.cur.execute(query, (embedding_text, doc_id, dataset_name))
        self.conn.commit() 