"""
Comprehensive test suite for the Advanced RAG Chatbot
"""
import os
import sys
import pytest
import asyncio
import tempfile
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules to test
from config import config, Config
from vector_store import Document, ChromaVectorStore, FAISSVectorStore, HybridVectorStore
from document_processor import DocumentProcessor, DocumentMetadata
from ml_utils import SemanticCache, QueryExpander, DocumentReranker, QueryAnalyzer
from rag_engine import RAGEngine, RAGConfig, GeminiClient

class TestConfig:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test config initialization"""
        test_config = Config()
        assert test_config.app_name == "Advanced RAG Chatbot"
        assert test_config.version == "2.0.0"
        assert test_config.model.embedding_dimension == 384
    
    def test_config_validation(self):
        """Test config validation"""
        # Test with missing API key
        with patch.dict(os.environ, {}, clear=True):
            test_config = Config()
            assert not test_config.validate_config()
    
    def test_model_config_retrieval(self):
        """Test model configuration retrieval"""
        test_config = Config()
        gemini_config = test_config.get_model_config("gemini")
        assert "model" in gemini_config
        assert gemini_config["model"] == "gemini-pro"

class TestVectorStore:
    """Test vector store implementations"""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                id="doc1",
                content="This is a test document about machine learning.",
                metadata={"title": "ML Test", "source": "test"}
            ),
            Document(
                id="doc2", 
                content="This document discusses natural language processing.",
                metadata={"title": "NLP Test", "source": "test"}
            )
        ]
    
    def test_document_creation(self):
        """Test document creation"""
        doc = Document(
            id="test_id",
            content="test content",
            metadata={"key": "value"}
        )
        assert doc.id == "test_id"
        assert doc.content == "test content"
        assert doc.metadata["key"] == "value"
    
    def test_chroma_vector_store_initialization(self):
        """Test ChromaDB vector store initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('chromadb.PersistentClient'):
                store = ChromaVectorStore()
                assert store.dimension > 0
                assert store.embedder is not None
    
    def test_faiss_vector_store_initialization(self):
        """Test FAISS vector store initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FAISSVectorStore(index_path=os.path.join(temp_dir, "test_index"))
            assert store.dimension > 0
            assert store.index is not None
    
    def test_embedding_generation(self):
        """Test embedding generation"""
        with patch('chromadb.PersistentClient'):
            store = ChromaVectorStore()
            embedding = store.embed_text("test text")
            assert len(embedding) == store.dimension
            assert isinstance(embedding, np.ndarray)

class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create document processor instance"""
        return DocumentProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.chunk_size > 0
        assert processor.chunk_overlap >= 0
        assert processor.supported_formats is not None
    
    def test_simple_chunking(self, processor):
        """Test simple text chunking"""
        text = "This is a long text. " * 100  # Create long text
        chunks = processor._simple_chunking(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= processor.chunk_size + 100 for chunk in chunks)  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_text_content_extraction(self, processor):
        """Test text file content extraction"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content for extraction.")
            f.flush()
            
            try:
                content = await processor._extract_text_content(Path(f.name))
                assert content == "This is test content for extraction."
            finally:
                os.unlink(f.name)
    
    def test_metadata_extraction(self, processor):
        """Test metadata extraction"""
        content = "This is a test document about artificial intelligence and machine learning."
        
        # Mock file path
        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            file_path = Path(f.name)
            
            # Run metadata extraction
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                metadata = loop.run_until_complete(
                    processor._extract_metadata(file_path, content)
                )
                
                assert metadata.filename == file_path.name
                assert metadata.word_count > 0
                assert metadata.character_count == len(content)
                assert metadata.quality_score is not None
            finally:
                loop.close()

class TestMLUtils:
    """Test ML utility functions"""
    
    def test_semantic_cache_initialization(self):
        """Test semantic cache initialization"""
        cache = SemanticCache(max_size=100, ttl=3600)
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert len(cache.cache) == 0
    
    def test_semantic_cache_operations(self):
        """Test cache set/get operations"""
        cache = SemanticCache(max_size=10, ttl=3600)
        
        # Test set/get
        cache.set("test query", "test response")
        result = cache.get("test query")
        assert result == "test response"
        
        # Test semantic similarity (should find similar query)
        similar_result = cache.get("test question", similarity_threshold=0.3)
        # This might be None if embeddings are too different
    
    def test_query_analyzer_initialization(self):
        """Test query analyzer initialization"""
        analyzer = QueryAnalyzer()
        assert analyzer.sentiment_analyzer is not None
    
    def test_query_analysis(self):
        """Test query analysis functionality"""
        analyzer = QueryAnalyzer()
        query = "What is machine learning and how does it work?"
        
        analysis = analyzer.analyze_query(query)
        assert analysis.intent is not None
        assert analysis.question_type is not None
        assert 0 <= analysis.confidence <= 1
        assert isinstance(analysis.topics, list)
    
    def test_document_reranker_initialization(self):
        """Test document reranker initialization"""
        with patch('sentence_transformers.CrossEncoder'):
            reranker = DocumentReranker()
            assert reranker.cross_encoder is not None

class TestRAGEngine:
    """Test RAG engine functionality"""
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create mock Gemini client"""
        client = Mock(spec=GeminiClient)
        client.generate_response = AsyncMock(return_value={
            'text': 'This is a test response',
            'finish_reason': 'STOP',
            'safety_ratings': [],
            'tokens_used': 50
        })
        return client
    
    @pytest.fixture
    def rag_config(self):
        """Create test RAG configuration"""
        return RAGConfig(
            model_name="gemini-pro",
            temperature=0.7,
            max_tokens=1000,
            top_k=3,
            enable_reranking=False,
            enable_query_expansion=False,
            enable_caching=False
        )
    
    def test_rag_config_creation(self, rag_config):
        """Test RAG configuration creation"""
        assert rag_config.model_name == "gemini-pro"
        assert rag_config.temperature == 0.7
        assert rag_config.top_k == 3
    
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    def test_rag_engine_initialization(self, rag_config):
        """Test RAG engine initialization"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                engine = RAGEngine(api_key='test_key', rag_config=rag_config)
                assert engine.api_key == 'test_key'
                assert engine.config == rag_config
                assert engine.gemini_client is not None
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    async def test_query_processing(self, mock_gemini_client, rag_config):
        """Test query processing"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                # Create engine
                engine = RAGEngine(api_key='test_key', rag_config=rag_config)
                engine.gemini_client = mock_gemini_client
                
                # Mock vector store
                engine.vector_store = Mock()
                engine.vector_store.similarity_search.return_value = [
                    Document(
                        id="test_doc",
                        content="Test document content",
                        metadata={"source": "test"},
                        score=0.9
                    )
                ]
                
                # Test query
                response = await engine.query("What is machine learning?")
                
                assert response.answer is not None
                assert response.confidence > 0
                assert response.processing_time > 0
                assert len(response.sources) > 0
    
    @pytest.mark.asyncio
    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    async def test_document_addition(self, rag_config):
        """Test adding documents to knowledge base"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                engine = RAGEngine(api_key='test_key', rag_config=rag_config)
                
                # Mock vector store and document processor
                engine.vector_store = Mock()
                engine.vector_store.add_documents = Mock()
                engine.document_processor = Mock()
                engine.document_processor.process_multiple_files = AsyncMock(return_value=[
                    Document(id="test", content="test content", metadata={})
                ])
                
                # Test adding documents
                result = await engine.add_documents(['test.txt'])
                
                assert result['success'] is True
                assert result['documents_added'] > 0

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_workflow(self):
        """Test complete workflow from document upload to query response"""
        # This test requires actual API key and is marked as integration
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            pytest.skip("GEMINI_API_KEY not available for integration test")
        
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Machine learning is a subset of artificial intelligence.")
            test_file = f.name
        
        try:
            # Initialize RAG engine
            config = RAGConfig(
                enable_reranking=False,
                enable_query_expansion=False,
                enable_caching=False
            )
            engine = RAGEngine(api_key=api_key, rag_config=config)
            
            # Add document
            result = await engine.add_documents([test_file])
            assert result['success'] is True
            
            # Query the document
            response = await engine.query("What is machine learning?")
            assert response.answer is not None
            assert response.confidence > 0
            
        finally:
            os.unlink(test_file)
    
    def test_configuration_consistency(self):
        """Test configuration consistency across modules"""
        # Ensure all modules use the same configuration
        from config import config as global_config
        
        assert global_config.model.chunk_size > 0
        assert global_config.vector_store.similarity_threshold > 0
        assert global_config.ml.response_time_threshold > 0
    
    def test_error_handling(self):
        """Test error handling across the system"""
        # Test with invalid API key
        with pytest.raises(Exception):
            RAGEngine(api_key="invalid_key")
        
        # Test with invalid configuration
        invalid_config = RAGConfig(chunk_size=-1)  # Invalid value
        # Should handle gracefully

class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.performance
    def test_embedding_performance(self):
        """Test embedding generation performance"""
        import time
        from vector_store import ChromaVectorStore
        
        with patch('chromadb.PersistentClient'):
            store = ChromaVectorStore()
            
            texts = [f"This is test text number {i}" for i in range(100)]
            
            start_time = time.time()
            embeddings = store.embed_documents(texts)
            end_time = time.time()
            
            # Should process 100 texts in reasonable time (< 10 seconds)
            assert end_time - start_time < 10
            assert len(embeddings) == 100
    
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test cache performance with many operations"""
        cache = SemanticCache(max_size=1000)
        
        # Add many items
        for i in range(100):
            cache.set(f"query_{i}", f"response_{i}")
        
        # Test retrieval performance
        import time
        start_time = time.time()
        
        for i in range(100):
            result = cache.get(f"query_{i}")
            assert result == f"response_{i}"
        
        end_time = time.time()
        
        # Should be very fast (< 1 second for 100 lookups)
        assert end_time - start_time < 1.0

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_document_handling(self):
        """Test handling of empty documents"""
        processor = DocumentProcessor()
        
        # Test with empty content
        chunks = processor._simple_chunking("")
        assert len(chunks) == 0
        
        # Test with very short content
        chunks = processor._simple_chunking("Hi")
        assert len(chunks) <= 1
    
    def test_large_document_handling(self):
        """Test handling of very large documents"""
        processor = DocumentProcessor()
        
        # Create very large text
        large_text = "This is a test sentence. " * 10000
        chunks = processor._simple_chunking(large_text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= processor.chunk_size + 100 for chunk in chunks)
    
    def test_special_character_handling(self):
        """Test handling of special characters"""
        processor = DocumentProcessor()
        
        # Text with special characters
        special_text = "This has émojis 🎉 and spëcial cháracters!"
        chunks = processor._simple_chunking(special_text)
        
        assert len(chunks) >= 1
        assert special_text in "".join(chunks)
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling of concurrent queries"""
        if not os.getenv('GEMINI_API_KEY'):
            pytest.skip("API key required for concurrent query test")
        
        config = RAGConfig(enable_caching=True)
        engine = RAGEngine(api_key=os.getenv('GEMINI_API_KEY'), rag_config=config)
        
        # Create multiple concurrent queries
        queries = [
            "What is AI?",
            "What is machine learning?", 
            "What is deep learning?",
            "What is neural network?",
            "What is data science?"
        ]
        
        # Execute concurrently
        tasks = [engine.query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all responses are valid (not exceptions)
        for response in responses:
            assert not isinstance(response, Exception)
            if hasattr(response, 'answer'):
                assert response.answer is not None

# Test fixtures for pytest
@pytest.fixture(scope="session")
def test_api_key():
    """Get test API key from environment"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        pytest.skip("No API key available for testing")
    return api_key

@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_text_file():
    """Create sample text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample document for testing the RAG system. "
                "It contains information about machine learning, "
                "artificial intelligence, and natural language processing.")
        f.flush()
        yield f.name
    os.unlink(f.name)

# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )

# Parametrized tests
@pytest.mark.parametrize("chunk_size,overlap", [
    (256, 50),
    (512, 100), 
    (1024, 200),
])
def test_chunking_parameters(chunk_size, overlap):
    """Test different chunking parameters"""
    processor = DocumentProcessor()
    processor.chunk_size = chunk_size
    processor.chunk_overlap = overlap
    
    # Create test text
    text = "This is a test sentence. " * 200
    chunks = processor._simple_chunking(text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= chunk_size + 100  # Allow some variance

@pytest.mark.parametrize("file_type,content", [
    (".txt", "Plain text content"),
    (".md", "# Markdown content\nWith some text"),
    (".html", "<html><body>HTML content</body></html>"),
    (".json", '{"key": "value", "text": "JSON content"}'),
])
def test_different_file_types(file_type, content):
    """Test processing different file types"""
    processor = DocumentProcessor()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=file_type, delete=False) as f:
        f.write(content)
        f.flush()
        
        try:
            # Test file type detection
            assert f.name.endswith(file_type)
            
            # Test content extraction (would need async context in real test)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                extracted = loop.run_until_complete(
                    processor._extract_content(Path(f.name))
                )
                assert extracted is not None
                assert len(extracted) > 0
            finally:
                loop.close()
                
        finally:
            os.unlink(f.name)

# Mock generators for testing
def generate_mock_documents(count: int = 10):
    """Generate mock documents for testing"""
    documents = []
    for i in range(count):
        doc = Document(
            id=f"mock_doc_{i}",
            content=f"This is mock document number {i} with test content about topic {i}.",
            metadata={
                "title": f"Mock Document {i}",
                "source": "test_generator",
                "category": f"category_{i % 3}"
            },
            score=0.8 + (i * 0.01)  # Varying scores
        )
        documents.append(doc)
    return documents

def generate_mock_queries(count: int = 5):
    """Generate mock queries for testing"""
    queries = [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Explain natural language processing",
        "What are neural networks?", 
        "Define deep learning algorithms"
    ]
    return queries[:count]

# Utility functions for tests
def assert_valid_embedding(embedding, expected_dim: int = 384):
    """Assert that an embedding is valid"""
    assert isinstance(embedding, np.ndarray)
    assert len(embedding) == expected_dim
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()

def assert_valid_response(response):
    """Assert that a RAG response is valid"""
    assert hasattr(response, 'answer')
    assert hasattr(response, 'confidence')
    assert hasattr(response, 'processing_time')
    assert hasattr(response, 'sources')
    
    assert response.answer is not None
    assert 0 <= response.confidence <= 1
    assert response.processing_time > 0
    assert isinstance(response.sources, list)

def assert_valid_document(document):
    """Assert that a document is valid"""
    assert hasattr(document, 'content')
    assert hasattr(document, 'metadata')
    assert hasattr(document, 'id')
    
    assert document.content is not None
    assert len(document.content) > 0
    assert isinstance(document.metadata, dict)

# Main test execution
if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not integration",  # Skip integration tests by default
        "--durations=10"  # Show slowest 10 tests
    ])

# Import required for some tests
try:
    import numpy as np
except ImportError:
    np = None
    pytest.skip("NumPy not available", allow_module_level=True)