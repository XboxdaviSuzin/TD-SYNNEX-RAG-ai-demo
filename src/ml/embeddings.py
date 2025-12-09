"""
Advanced Embedding Engine
Supports PyTorch (sentence-transformers), HuggingFace, and TensorFlow
Includes INT8 quantization for production optimization
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingBackend(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    HUGGINGFACE = "huggingface"
    MOCK = "mock"  # For testing without GPU


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding generation"""
    model_name: str
    backend: str
    embedding_dim: int
    avg_latency_ms: float
    quantized: bool
    memory_mb: float
    throughput_docs_per_sec: float


class EmbeddingEngine:
    """
    Production-grade embedding engine with multiple backend support
    - PyTorch: sentence-transformers (all-MiniLM-L6-v2)
    - TensorFlow: Universal Sentence Encoder
    - HuggingFace: transformers pipeline
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        backend: EmbeddingBackend = EmbeddingBackend.PYTORCH,
        use_quantization: bool = False,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.backend = backend
        self.use_quantization = use_quantization
        self.device = device
        self.model = None
        self.embedding_dim = 384  # Default for MiniLM
        self._latencies: List[float] = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on backend"""
        logger.info(f"Initializing {self.backend.value} embedding model: {self.model_name}")
        
        if self.backend == EmbeddingBackend.PYTORCH:
            self._init_pytorch()
        elif self.backend == EmbeddingBackend.TENSORFLOW:
            self._init_tensorflow()
        elif self.backend == EmbeddingBackend.HUGGINGFACE:
            self._init_huggingface()
        else:
            self._init_mock()
    
    def _init_pytorch(self):
        """Initialize PyTorch sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            if self.use_quantization:
                # Apply dynamic INT8 quantization
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("Applied INT8 quantization to PyTorch model")
            
            logger.info(f"PyTorch model loaded: dim={self.embedding_dim}")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to mock")
            self._init_mock()
    
    def _init_tensorflow(self):
        """Initialize TensorFlow Universal Sentence Encoder"""
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # Use Universal Sentence Encoder
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            self.embedding_dim = 512
            logger.info(f"TensorFlow USE model loaded: dim={self.embedding_dim}")
        except (ImportError, Exception) as e:
            logger.warning(f"TensorFlow model not available: {e}, falling back to mock")
            self._init_mock()
    
    def _init_huggingface(self):
        """Initialize HuggingFace transformers pipeline"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{self.model_name}")
            self.model = AutoModel.from_pretrained(f"sentence-transformers/{self.model_name}")
            self.model.to(self.device)
            
            # Get embedding dimension from model config
            self.embedding_dim = self.model.config.hidden_size
            
            if self.use_quantization:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            logger.info(f"HuggingFace model loaded: dim={self.embedding_dim}")
        except ImportError:
            logger.warning("transformers not available, falling back to mock")
            self._init_mock()
    
    def _init_mock(self):
        """Initialize mock embeddings for testing"""
        self.backend = EmbeddingBackend.MOCK
        self.embedding_dim = 384
        logger.info("Using mock embeddings for testing")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for input texts
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        start_time = time.time()
        
        if self.backend == EmbeddingBackend.PYTORCH and self.model is not None:
            embeddings = self._encode_pytorch(texts, batch_size, show_progress)
        elif self.backend == EmbeddingBackend.TENSORFLOW and self.model is not None:
            embeddings = self._encode_tensorflow(texts)
        elif self.backend == EmbeddingBackend.HUGGINGFACE and self.model is not None:
            embeddings = self._encode_huggingface(texts, batch_size)
        else:
            embeddings = self._encode_mock(texts)
        
        latency = (time.time() - start_time) * 1000
        self._latencies.append(latency / len(texts))
        
        return embeddings
    
    def _encode_pytorch(self, texts: List[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """Encode using PyTorch sentence-transformers"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def _encode_tensorflow(self, texts: List[str]) -> np.ndarray:
        """Encode using TensorFlow USE"""
        embeddings = self.model(texts)
        return embeddings.numpy()
    
    def _encode_huggingface(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode using HuggingFace transformers"""
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _encode_mock(self, texts: List[str]) -> np.ndarray:
        """Generate deterministic mock embeddings"""
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            embeddings.append(np.random.randn(self.embedding_dim).astype(np.float32))
        return np.array(embeddings)
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Get embedding engine metrics"""
        import sys
        
        avg_latency = np.mean(self._latencies) if self._latencies else 0.0
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        # Estimate memory usage
        memory_mb = sys.getsizeof(self.model) / (1024 * 1024) if self.model else 0.0
        
        return EmbeddingMetrics(
            model_name=self.model_name,
            backend=self.backend.value,
            embedding_dim=self.embedding_dim,
            avg_latency_ms=round(avg_latency, 2),
            quantized=self.use_quantization,
            memory_mb=round(memory_mb, 2),
            throughput_docs_per_sec=round(throughput, 2)
        )


def get_embedding_engine(
    backend: str = "pytorch",
    quantized: bool = False
) -> EmbeddingEngine:
    """Factory function to get embedding engine"""
    backend_map = {
        "pytorch": EmbeddingBackend.PYTORCH,
        "tensorflow": EmbeddingBackend.TENSORFLOW,
        "huggingface": EmbeddingBackend.HUGGINGFACE,
        "mock": EmbeddingBackend.MOCK
    }
    
    return EmbeddingEngine(
        model_name="all-MiniLM-L6-v2",
        backend=backend_map.get(backend, EmbeddingBackend.MOCK),
        use_quantization=quantized
    )


if __name__ == "__main__":
    # Test embedding engine
    engine = get_embedding_engine("mock")
    
    texts = [
        "Cisco Catalyst 9300 switch for enterprise networks",
        "HP ProLiant DL380 server with dual Xeon processors",
        "Dell PowerEdge R750 rack server"
    ]
    
    embeddings = engine.encode(texts)
    print(f"Generated embeddings: {embeddings.shape}")
    print(f"Metrics: {engine.get_metrics()}")
