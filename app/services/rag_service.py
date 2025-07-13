from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
import joblib
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from app.services.preprocessor.embedding_preprocessing_strategy import EmbeddingPreprocessingStrategy
from app.services.preprocessor.preprocessing_service import PreprocessingService
from app.models.enums import DatasetName, ModelType

class RAGService:
    def __init__(self):
        self.preprocessor = PreprocessingService(EmbeddingPreprocessingStrategy())
        self.index = None
        self.doc_ids = []
        self.documents = []
        self.current_dataset = None
        
        self.llm = None
        self.tokenizer = None
        
    def _initialize_llm(self):
        """Initialize the language model for text generation"""
        if self.llm is not None:
            return
            
        try:
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            if torch.cuda.is_available():
                self.llm = self.llm.to('cuda')
                
            print(f"LLM initialized: {model_name}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM: {e}")
            print("Falling back to template-based generation")
            self.llm = None
        
    def initialize_from_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Initialize RAG using existing BERT embeddings from a dataset"""
        print(f"Initializing RAG from existing BERT embeddings: {dataset_name}")
        
        try:
            from app.services.model_manager import model_manager
            dataset_enum = DatasetName(dataset_name)
            bert_assets = model_manager.get_model_assets(dataset_enum, ModelType.BERT)
            
            if not bert_assets:
                raise ValueError(f"BERT model not found for dataset: {dataset_name}")
            
            stored_embeddings = bert_assets["doc_embeddings"]
            stored_doc_ids = bert_assets["doc_ids"]
            
            if stored_embeddings is None:
                raise ValueError(f"BERT embeddings not found for dataset: {dataset_name}")
            
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            doc_data = db_service.get_documents_by_ids(dataset_name, stored_doc_ids)
            
            if len(doc_data) != len(stored_doc_ids):
                print(f"Warning: Only {len(doc_data)} documents found in database out of {len(stored_doc_ids)} embeddings")
            
            doc_texts = []
            valid_indices = []
            
            for i, doc in enumerate(doc_data):
                if doc and 'text' in doc:
                    cleaned_text = self._clean_document(doc['text'])
                    if cleaned_text:
                        doc_texts.append(cleaned_text)
                        valid_indices.append(i)
            
            filtered_embeddings = stored_embeddings[valid_indices]
            filtered_doc_ids = [stored_doc_ids[i] for i in valid_indices]
            
            dimension = filtered_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            faiss.normalize_L2(filtered_embeddings)
            
            self.index.add(filtered_embeddings.astype('float32'))
            
            self.documents = doc_texts
            self.doc_ids = filtered_doc_ids
            self.current_dataset = dataset_name
            
            self._save_rag_metadata()
            
            return {
                "num_documents": len(doc_texts),
                "embedding_dimension": dimension,
                "index_size": self.index.ntotal,
                "embeddings_source": "existing_bert_model",
                "dataset_name": dataset_name,
                "bert_model_used": "all-MiniLM-L6-v2"
            }
            
        except Exception as e:
            print(f"Error initializing RAG from BERT embeddings: {e}")
            raise e
    
    def _clean_document(self, text: str) -> str:
        """Clean and normalize document text"""
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text.split()) < 3:
            return ""
            
        return text
    
    def _save_rag_metadata(self):
        """Save RAG metadata (not embeddings - they're in BERT!)"""
        rag_dir = "app/saved_models/rag"
        os.makedirs(rag_dir, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(rag_dir, "faiss_index.bin"))
        
        joblib.dump(self.doc_ids, os.path.join(rag_dir, "doc_ids.joblib"))
        joblib.dump(self.documents, os.path.join(rag_dir, "documents.joblib"))
        joblib.dump(self.current_dataset, os.path.join(rag_dir, "dataset.joblib"))
        
        print(f"RAG metadata saved to {rag_dir} (embeddings remain in BERT model)")
    
    def load_vector_store(self) -> bool:
        """Load RAG vector store from disk"""
        rag_dir = "app/saved_models/rag"
        
        try:
            self.index = faiss.read_index(os.path.join(rag_dir, "faiss_index.bin"))
            
            self.doc_ids = joblib.load(os.path.join(rag_dir, "doc_ids.joblib"))
            self.documents = joblib.load(os.path.join(rag_dir, "documents.joblib"))
            self.current_dataset = joblib.load(os.path.join(rag_dir, "dataset.joblib"))
            
            print(f"RAG vector store loaded with {len(self.documents)} documents")
            print(f"Using embeddings from BERT model for dataset: {self.current_dataset}")
            return True
            
        except FileNotFoundError:
            print("RAG vector store not found. Please create it first.")
            return False
        except Exception as e:
            print(f"Error loading RAG vector store: {e}")
            return False
    
    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve relevant documents using existing BERT embeddings"""
        if self.index is None:
            raise ValueError("Vector store not initialized. Please create it first.")
        
        processed_query = self.preprocessor.normalize(query)
        
        from app.services.model_manager import model_manager
        dataset_enum = DatasetName(self.current_dataset)
        bert_assets = model_manager.get_model_assets(dataset_enum, ModelType.BERT)
        
        if not bert_assets or bert_assets["model"] is None:
            raise ValueError("BERT model not available for query embedding")
        
        bert_model = bert_assets["model"]
        query_embedding = bert_model.encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc_id = self.doc_ids[idx]
                doc_text = self.documents[idx]
                results.append((doc_id, doc_text, float(score)))
        
        return results
    
    def _calculate_relevance_score(self, query: str, document: str) -> float:
        """Calculate relevance score between query and document"""
        return 0.0
    
    def _create_prompt(self, query: str, context_docs: List[str]) -> str:
        """Create a prompt for the language model"""
        context_text = "\n\n".join(context_docs)
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'factual'
        elif any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus']):
            return 'comparative'
        elif any(word in query_lower for word in ['explain', 'describe', 'tell me about']):
            return 'descriptive'
        else:
            return 'general'
    
    def _generate_with_llm(self, prompt: str, max_length: int = 200) -> str:
        """Generate answer using the language model"""
        if self.llm is None:
            return self._template_based_generation(prompt, max_length)
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            answer = response[len(prompt):].strip()
            
            return self._clean_answer(answer)
            
        except Exception as e:
            print(f"Error generating with LLM: {e}")
            return self._template_based_generation(prompt, max_length)
    
    def _template_based_generation(self, prompt: str, max_length: int) -> str:
        """Generate answer using template-based approach"""
        try:
            lines = prompt.split('\n')
            context_start = None
            question_start = None
            
            for i, line in enumerate(lines):
                if line.startswith('Context:'):
                    context_start = i + 1
                elif line.startswith('Question:'):
                    question_start = i
                    break
            
            if context_start is None or question_start is None:
                return "I cannot process this question properly."
            
            context_lines = lines[context_start:question_start]
            question = lines[question_start].replace('Question:', '').strip()
            
            context_text = ' '.join(context_lines)
            
            question_words = question.lower().split()
            context_lower = context_text.lower()
            
            relevant_sentences = []
            sentences = context_text.split('. ')
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in question_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer = '. '.join(relevant_sentences[:3])
                if not answer.endswith('.'):
                    answer += '.'
                return answer
            else:
                answer = '. '.join(sentences[:2])
                if not answer.endswith('.'):
                    answer += '.'
                return answer
                
        except Exception as e:
            print(f"Error in template generation: {e}")
            return "I cannot answer this question based on the provided context."
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        answer = re.sub(r'\s+', ' ', answer.strip())
        
        answer = re.sub(r'^Answer:\s*', '', answer)
        answer = re.sub(r'^Based on the context,\s*', '', answer)
        
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer
    
    def generate_answer(self, query: str, context_docs: List[str], max_length: int = 200) -> str:
        """Generate an answer based on the query and context documents"""
        if not context_docs:
            return "I cannot answer this question as no relevant documents were found."
        
        prompt = self._create_prompt(query, context_docs)
        
        answer = self._generate_with_llm(prompt, max_length)
        
        return answer
    
    def rag_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform RAG search using existing BERT embeddings"""
        try:
            retrieved_docs = self.retrieve_relevant_documents(query, top_k)
            
            if not retrieved_docs:
                return {
                    "query": query,
                    "answer": "I cannot answer this question as no relevant documents were found.",
                    "retrieved_documents": [],
                    "context_used": 0
                }
            
            context_docs = [doc_text for _, doc_text, _ in retrieved_docs]
            
            answer = self.generate_answer(query, context_docs)
            
            formatted_docs = []
            for doc_id, doc_text, score in retrieved_docs:
                formatted_docs.append({
                    "doc_id": doc_id,
                    "text": doc_text,
                    "score": score
                })
            
            return {
                "query": query,
                "answer": answer,
                "retrieved_documents": formatted_docs,
                "context_used": len(context_docs)
            }
            
        except Exception as e:
            print(f"Error in RAG search: {e}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "retrieved_documents": [],
                "context_used": 0
            }
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        if self.index is None:
            return {
                "status": "not_initialized",
                "message": "Vector store not created. Use create_vector_store_from_dataset first."
            }
        
        return {
            "status": "loaded",
            "num_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "dataset": self.current_dataset,
            "embeddings_source": "existing_bert_model",
            "bert_model": "all-MiniLM-L6-v2"
        }
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available"""
        return self.llm is not None 