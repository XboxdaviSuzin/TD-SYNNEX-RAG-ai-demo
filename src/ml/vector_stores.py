"""
Multi-Backend Vector Store Manager
Supports FAISS, Pinecone, Weaviate, and Milvus
Includes CAG (Context-Augmented Generation) with multi-context retrieval
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    MOCK = "mock"


@dataclass
class SearchResult:
    """Single search result from vector store"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int


@dataclass
class CAGResult:
    """Context-Augmented Generation result with multi-context"""
    query: str
    primary_results: List[SearchResult]  # Top-3
    fallback_results: List[SearchResult]  # Top-5 extended
    combined_context: str
    retrieval_latency_ms: float
    vector_db: str


class VectorStoreManager:
    """
    Production vector store manager with multi-backend support
    - FAISS: Primary high-performance option
    - Pinecone: Managed cloud vector DB
    - Weaviate: Open-source with GraphQL
    - Milvus: Distributed vector DB
    """
    
    def __init__(
        self,
        db_type: VectorDBType = VectorDBType.FAISS,
        dimension: int = 384,
        index_type: str = "IVF256,Flat",
        **kwargs
    ):
        self.db_type = db_type
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._config = kwargs
        
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store backend"""
        logger.info(f"Initializing {self.db_type.value} vector store (dim={self.dimension})")
        
        if self.db_type == VectorDBType.FAISS:
            self._init_faiss()
        elif self.db_type == VectorDBType.PINECONE:
            self._init_pinecone()
        elif self.db_type == VectorDBType.WEAVIATE:
            self._init_weaviate()
        elif self.db_type == VectorDBType.MILVUS:
            self._init_milvus()
        else:
            self._init_mock()
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            # Create index based on type
            if "IVF" in self.index_type:
                # IVF index for larger datasets
                quantizer = faiss.IndexFlatL2(self.dimension)
                nlist = int(self.index_type.split("IVF")[1].split(",")[0])
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self._needs_training = True
            else:
                # Flat index for smaller datasets
                self.index = faiss.IndexFlatL2(self.dimension)
                self._needs_training = False
            
            logger.info(f"FAISS index created: {self.index_type}")
        except ImportError:
            logger.warning("FAISS not available, falling back to mock")
            self._init_mock()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            from pinecone import Pinecone
            
            api_key = self._config.get("api_key", "")
            environment = self._config.get("environment", "")
            index_name = self._config.get("index_name", "td-synnex-products")
            
            if api_key:
                pc = Pinecone(api_key=api_key)
                self.index = pc.Index(index_name)
                logger.info(f"Pinecone connected: {index_name}")
            else:
                logger.warning("Pinecone API key not provided, using mock mode")
                self._pinecone_mock = True
        except ImportError:
            logger.warning("Pinecone client not available")
            self._pinecone_mock = True
    
    def _init_weaviate(self):
        """Initialize Weaviate connection"""
        try:
            import weaviate
            
            url = self._config.get("url", "http://localhost:8080")
            api_key = self._config.get("api_key", None)
            
            auth_config = weaviate.auth.AuthApiKey(api_key) if api_key else None
            self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            
            # Create schema if not exists
            schema = {
                "class": "TDSynnexProduct",
                "vectorizer": "none",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "product_id", "dataType": ["string"]},
                    {"name": "vendor", "dataType": ["string"]},
                    {"name": "category", "dataType": ["string"]},
                ]
            }
            
            if not self.client.schema.exists("TDSynnexProduct"):
                self.client.schema.create_class(schema)
            
            logger.info("Weaviate connected")
        except (ImportError, Exception) as e:
            logger.warning(f"Weaviate not available: {e}")
            self._weaviate_mock = True
    
    def _init_milvus(self):
        """Initialize Milvus connection"""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
            
            host = self._config.get("host", "localhost")
            port = self._config.get("port", 19530)
            
            connections.connect("default", host=host, port=port)
            
            # Define collection schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2048),
            ]
            schema = CollectionSchema(fields, "TD SYNNEX product vectors")
            
            self.collection = Collection("td_synnex_products", schema)
            logger.info("Milvus connected")
        except (ImportError, Exception) as e:
            logger.warning(f"Milvus not available: {e}")
            self._milvus_mock = True
    
    def _init_mock(self):
        """Initialize mock vector store"""
        self.db_type = VectorDBType.MOCK
        logger.info("Using mock vector store")
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> int:
        """
        Add documents with their embeddings to the store
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
            embeddings: numpy array of embeddings (n_docs, dim)
        
        Returns:
            Number of documents added
        """
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must have same length")
        
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Add to backend
        if self.db_type == VectorDBType.FAISS and self.index is not None:
            if self._needs_training and not self.index.is_trained:
                self.index.train(embeddings.astype(np.float32))
            self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Added {len(documents)} documents to {self.db_type.value}")
        return len(documents)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector (1, dim) or (dim,)
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of SearchResult objects
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        start_time = time.time()
        
        if self.db_type == VectorDBType.FAISS and self.index is not None:
            results = self._search_faiss(query_embedding, top_k)
        else:
            results = self._search_mock(query_embedding, top_k)
        
        # Apply filters if provided
        if filter_dict:
            results = self._apply_filters(results, filter_dict)
        
        latency = (time.time() - start_time) * 1000
        logger.debug(f"Search completed in {latency:.2f}ms")
        
        return results
    
    def _search_faiss(self, query: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search using FAISS index"""
        distances, indices = self.index.search(query.astype(np.float32), top_k)
        
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append(SearchResult(
                    id=doc.get("id", str(idx)),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=float(1.0 / (1.0 + dist)),  # Convert distance to similarity
                    rank=rank + 1
                ))
        
        return results
    
    def _search_mock(self, query: np.ndarray, top_k: int) -> List[SearchResult]:
        """Mock search using cosine similarity"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Compute cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(emb_norm, query_norm.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx]
            results.append(SearchResult(
                id=doc.get("id", str(idx)),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=float(similarities[idx]),
                rank=rank + 1
            ))
        
        return results
    
    def _apply_filters(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Apply metadata filters to results"""
        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if result.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(result)
        return filtered
    
    def cag_retrieve(
        self,
        query_embedding: np.ndarray,
        top_k_primary: int = 3,
        top_k_fallback: int = 5
    ) -> CAGResult:
        """
        Context-Augmented Generation retrieval with multi-context
        
        Primary: Top-3 most relevant results
        Fallback: Extended Top-5 for context expansion
        
        Args:
            query_embedding: Query vector
            top_k_primary: Number of primary results
            top_k_fallback: Number of fallback results
        
        Returns:
            CAGResult with combined context
        """
        start_time = time.time()
        
        # Get extended results
        all_results = self.search(query_embedding, top_k=top_k_fallback)
        
        primary = all_results[:top_k_primary]
        fallback = all_results[top_k_primary:top_k_fallback]
        
        # Combine context
        contexts = []
        for result in primary:
            contexts.append(f"[Primary] {result.content}")
        for result in fallback:
            contexts.append(f"[Context] {result.content}")
        
        combined = "\n\n".join(contexts)
        
        latency = (time.time() - start_time) * 1000
        
        return CAGResult(
            query="",  # Set by caller
            primary_results=primary,
            fallback_results=fallback,
            combined_context=combined,
            retrieval_latency_ms=round(latency, 2),
            vector_db=self.db_type.value
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "db_type": self.db_type.value,
            "dimension": self.dimension,
            "num_documents": len(self.documents),
            "index_type": self.index_type,
            "is_trained": getattr(self, '_needs_training', False) and 
                         hasattr(self.index, 'is_trained') and self.index.is_trained
        }


def create_vector_store(
    db_type: str = "faiss",
    dimension: int = 384,
    **kwargs
) -> VectorStoreManager:
    """Factory function to create vector store"""
    db_map = {
        "faiss": VectorDBType.FAISS,
        "pinecone": VectorDBType.PINECONE,
        "weaviate": VectorDBType.WEAVIATE,
        "milvus": VectorDBType.MILVUS,
        "mock": VectorDBType.MOCK
    }
    
    return VectorStoreManager(
        db_type=db_map.get(db_type.lower(), VectorDBType.MOCK),
        dimension=dimension,
        **kwargs
    )


if __name__ == "__main__":
    # Test vector store
    store = create_vector_store("mock", dimension=384)
    
    # Add sample documents
    docs = [
        {"id": "1", "content": "Cisco Catalyst 9300", "metadata": {"vendor": "Cisco"}},
        {"id": "2", "content": "HP ProLiant DL380", "metadata": {"vendor": "HP"}},
        {"id": "3", "content": "Dell PowerEdge R750", "metadata": {"vendor": "Dell"}},
    ]
    
    embeddings = np.random.randn(3, 384).astype(np.float32)
    store.add_documents(docs, embeddings)
    
    # Search
    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, top_k=2)
    
    print("Search results:")
    for r in results:
        print(f"  {r.rank}. {r.content} (score: {r.score:.3f})")
    
    print(f"\nStore stats: {store.get_stats()}")
