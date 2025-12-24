"""
Vector Store implementation
"""
from datetime import date, datetime
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TF optimizations
os.environ['USE_TF'] = '0'  # Don't use TensorFlow backend
os.environ['USE_TORCH'] = '1'  # Use PyTorch backend only

import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import faiss
import chromadb
from chromadb.config import Settings
import pandas as pd
import time

from config import config, PERFORMANCE_METRICS
from ml_utils import SemanticCache, QueryExpander, DocumentReranker

logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    id: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None

class BaseVectorStore(ABC):
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or config.model.embedding_model
        self.embedder = SentenceTransformer(self.embedding_model)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.cache = SemanticCache() if config.ml.enable_semantic_caching else None
        
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        pass
    
    def embed_text(self, text: str) -> np.ndarray:
        if self.cache and self.cache.get(text) is not None:
            return self.cache.get(text)
            
        embedding = self.embedder.encode(text, normalize_embeddings=True)
        
        if self.cache:
            self.cache.set(text, embedding)
            
        return embedding
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)

class ChromaVectorStore(BaseVectorStore):
    
    def __init__(self, collection_name: str = "rag_documents", **kwargs):
        super().__init__(**kwargs)
        
        # Initialize ChromaDB
        persist_dir = config.database.chromadb_path
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB with collection: {collection_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        
        from datetime import datetime, date
        import json
        
        def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, (datetime, date)):
                    sanitized[key] = value.isoformat()
                elif value is None:
                    sanitized[key] = ""
                elif isinstance(value, (list, tuple)):
                    sanitized[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    sanitized[key] = json.dumps(value)
                elif isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                else:
                    sanitized[key] = str(value)
            return sanitized
    
        texts = [doc.content for doc in documents]
        embeddings = self.embed_documents(texts)
        
        ids = [doc.id or f"doc_{i}" for i, doc in enumerate(documents)]
        # Sanitize all metadata
        metadatas = [sanitize_metadata(doc.metadata) for doc in documents]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            score = 1 - distance  # Convert distance to similarity
            documents.append(Document(
                content=doc,
                metadata=metadata,
                id=results['ids'][0][i],
                score=score
            ))
        
        return documents
    
    def delete_documents(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from ChromaDB")

class FAISSVectorStore(BaseVectorStore):
    
    def __init__(self, index_path: str = None, **kwargs):
        super().__init__(**kwargs)
        
        self.index_path = index_path or config.database.faiss_index_path
        self.documents: Dict[str, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.is_trained = False
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"Initialized FAISS index with dimension: {self.dimension}")
    
    def _load_index(self) -> None:
        if os.path.exists(f"{self.index_path}.index"):
            try:
                self.index = faiss.read_index(f"{self.index_path}.index")
                
                with open(f"{self.index_path}.metadata", 'rb') as f:
                    metadata = pickle.load(f)
                    self.documents = metadata['documents']
                    self.id_to_index = metadata['id_to_index']
                    self.index_to_id = metadata['index_to_id']
                
                self.is_trained = True
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
    
    def _save_index(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        metadata = {
            'documents': self.documents,
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id
        }
        
        with open(f"{self.index_path}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("Saved FAISS index")
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        
        texts = [doc.content for doc in documents]
        embeddings = self.embed_documents(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Update mappings
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{start_idx + i}"
            idx = start_idx + i
            
            self.documents[doc_id] = doc
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
        
        self.is_trained = True
        self._save_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS index")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search in FAISS"""
        if not self.is_trained or self.index.ntotal == 0:
            return []
        
        query_embedding = self.embed_text(query).astype('float32').reshape(1, -1)
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc_id = self.index_to_id.get(idx)
            if doc_id and doc_id in self.documents:
                doc = self.documents[doc_id]
                doc.score = float(score)
                documents.append(doc)
        
        return documents
    
    def delete_documents(self, ids: List[str]) -> None:
        remaining_docs = []
        
        for doc_id, doc in self.documents.items():
            if doc_id not in ids:
                remaining_docs.append(doc)
        
        # Rebuild index
        self.documents.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.is_trained = False
        
        if remaining_docs:
            self.add_documents(remaining_docs)
        
        logger.info(f"Deleted {len(ids)} documents from FAISS index")

class HybridVectorStore(BaseVectorStore):
    
    def __init__(self, primary_store: str = "chromadb", **kwargs):
        super().__init__(**kwargs)
        
        self.primary_store = self._create_store(primary_store)
        self.secondary_store = self._create_store("faiss" if primary_store != "faiss" else "chromadb")
        
        self.reranker = DocumentReranker() if config.ml.enable_reranking else None
        self.query_expander = QueryExpander() if config.ml.enable_query_expansion else None
        
        # Clustering for document organization
        self.clusterer = KMeans(n_clusters=10, random_state=42)
        self.is_clustered = False
        
        logger.info(f"Initialized Hybrid Vector Store with {primary_store} as primary")
    
    def _create_store(self, store_type: str) -> BaseVectorStore:
        if store_type == "chromadb":
            return ChromaVectorStore()
        elif store_type == "faiss":
            return FAISSVectorStore()
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
            from datetime import datetime, date
            import json
    
        def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, (datetime, date)):
                    sanitized[key] = value.isoformat()
                elif value is None:
                    sanitized[key] = ""
                elif isinstance(value, (list, tuple)):
                    sanitized[key] = ", ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    sanitized[key] = json.dumps(value)
                elif isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                else:
                    sanitized[key] = str(value)
            return sanitized
        
        texts = [doc.content for doc in documents]
        embeddings = self.embed_documents(texts)
        
        ids = [doc.id or f"doc_{i}" for i, doc in enumerate(documents)]
        metadatas = [sanitize_metadata(doc.metadata) for doc in documents]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def _update_clustering(self) -> None:
        try:
            sample_docs = self.primary_store.similarity_search("", k=100)
            if len(sample_docs) > 10:
                embeddings = [doc.embedding for doc in sample_docs if doc.embedding is not None]
                if embeddings:
                    embeddings = np.array(embeddings)
                    self.clusterer.fit(embeddings)
                    self.is_clustered = True
                    logger.info("Updated document clustering")
        except Exception as e:
            logger.error(f"Error updating clustering: {e}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        start_time = time.time()
        
        expanded_queries = [query]
        if self.query_expander:
            expanded_queries.extend(self.query_expander.expand_query(query))
        
        all_documents = []
        for expanded_query in expanded_queries:
            primary_docs = self.primary_store.similarity_search(expanded_query, k=k*2)
            
            if len(primary_docs) < k:
                secondary_docs = self.secondary_store.similarity_search(expanded_query, k=k)
                primary_docs.extend(secondary_docs)
            
            all_documents.extend(primary_docs)
        
        seen_ids = set()
        unique_docs = []
        for doc in all_documents:
            if doc.id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc.id)
        
        unique_docs.sort(key=lambda x: x.score or 0, reverse=True)
        
        if self.reranker and len(unique_docs) > 1:
            unique_docs = self.reranker.rerank_documents(query, unique_docs[:k*2])
            
        response_time = time.time() - start_time
        PERFORMANCE_METRICS["response_time"].append(response_time)
        PERFORMANCE_METRICS["query_count"] += 1
        
        return unique_docs[:k]
    
    def delete_documents(self, ids: List[str]) -> None:
        self.primary_store.delete_documents(ids)
        self.secondary_store.delete_documents(ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "primary_store": type(self.primary_store).__name__,
            "secondary_store": type(self.secondary_store).__name__,
            "is_clustered": self.is_clustered,
            "embedding_dimension": self.dimension,
            "cache_enabled": self.cache is not None,
            "reranking_enabled": self.reranker is not None,
            "query_expansion_enabled": self.query_expander is not None
        }

def create_vector_store(store_type: str = None, **kwargs) -> BaseVectorStore:
    store_type = store_type or config.vector_store.default_store
    
    if store_type == "chromadb":
        return ChromaVectorStore(**kwargs)
    elif store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    elif store_type == "hybrid":
        return HybridVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")

__all__ = [
    "Document",
    "BaseVectorStore", 
    "ChromaVectorStore",
    "FAISSVectorStore", 
    "HybridVectorStore",
    "create_vector_store"
]
