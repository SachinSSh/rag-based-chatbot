"""
Advanced RAG Engine with Gemini API integration and ML enhancements
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import traceback

# Google Gemini imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Local imports
from config import config, SYSTEM_PROMPTS, PERFORMANCE_METRICS
from vector_store import Document, create_vector_store, BaseVectorStore
from document_processor import DocumentProcessor
from ml_utils import (
    QueryAnalyzer,
    QueryExpander,
    DocumentReranker,
    SemanticCache,
    PerformanceMonitor,
    QueryAnalysis,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG response with metadata"""

    answer: str
    sources: List[Document]
    query_analysis: QueryAnalysis
    confidence: float
    processing_time: float
    tokens_used: int
    cache_hit: bool
    error: Optional[str] = None


@dataclass
class RAGConfig:
    """RAG engine configuration"""

    model_name: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_k: int = 5
    similarity_threshold: float = 0.3
    enable_reranking: bool = True
    enable_query_expansion: bool = True
    enable_caching: bool = True
    system_prompt_type: str = "default"


class GeminiClient:
    """Enhanced Gemini API client with error handling and rate limiting"""

    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds

    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with error handling and rate limiting"""
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - time_since_last)

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 40),
                max_output_tokens=kwargs.get("max_tokens", 2048),
            )

            # Generate response
            self.last_request_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=self.safety_settings,
                ),
            )

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    return {
                        "text": candidate.content.parts[0].text,
                        "finish_reason": candidate.finish_reason,
                        "safety_ratings": candidate.safety_ratings,
                        "tokens_used": response.usage_metadata.total_token_count
                        if response.usage_metadata
                        else 0,
                    }

            raise Exception("No valid response generated")

        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            raise e


class RAGEngine:
    """Advanced RAG Engine with ML enhancements and performance optimization"""

    def __init__(self, api_key: str = None, rag_config: RAGConfig = None):
        self.api_key = api_key or config.gemini_api_key
        self.config = rag_config or RAGConfig()

        if not self.api_key:
            raise ValueError("Gemini API key is required")

        # Initialize components
        self.gemini_client = GeminiClient(self.api_key, self.config.model_name)
        self.vector_store: BaseVectorStore = None
        self.document_processor = DocumentProcessor()

        # ML components
        self.query_analyzer = QueryAnalyzer()
        self.query_expander = (
            QueryExpander() if self.config.enable_query_expansion else None
        )
        self.document_reranker = (
            DocumentReranker() if self.config.enable_reranking else None
        )
        self.semantic_cache = SemanticCache() if self.config.enable_caching else None
        self.performance_monitor = PerformanceMonitor()

        # System prompts
        self.system_prompts = SYSTEM_PROMPTS

        # Initialize vector store
        self._initialize_vector_store()

        logger.info(f"Initialized RAG Engine with model: {self.config.model_name}")

    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            self.vector_store = create_vector_store(
                store_type=config.vector_store.default_store
            )
            logger.info(f"Initialized vector store: {type(self.vector_store).__name__}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise e

    async def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the knowledge base"""
        start_time = time.time()

        try:
            # Process documents
            logger.info(f"Processing {len(file_paths)} documents...")
            documents = await self.document_processor.process_multiple_files(file_paths)

            if not documents:
                return {
                    "success": False,
                    "message": "No documents were successfully processed",
                    "metrics": self.document_processor.get_processing_metrics(),
                }

            # Add to vector store
            logger.info(f"Adding {len(documents)} documents to vector store...")
            await asyncio.get_event_loop().run_in_executor(
                None, self.vector_store.add_documents, documents
            )

            processing_time = time.time() - start_time
            metrics = self.document_processor.get_processing_metrics()

            logger.info(
                f"Successfully added {len(documents)} documents in {processing_time:.2f}s"
            )

            return {
                "success": True,
                "documents_added": len(documents),
                "processing_time": processing_time,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    async def query(self, query: str, **kwargs) -> RAGResponse:
        """Process query and generate response"""
        start_time = time.time()
        cache_hit = False
        error = None

        try:
            # Check cache first
            if self.semantic_cache:
                cached_response = self.semantic_cache.get(query)
                if cached_response:
                    cache_hit = True
                    cached_response.cache_hit = True
                    cached_response.processing_time = time.time() - start_time
                    return cached_response

            # Analyze query
            query_analysis = self.query_analyzer.analyze_query(query)

            # Expand query if enabled
            expanded_queries = [query]
            if self.query_expander and query_analysis.complexity > 0.3:
                try:
                    expansions = self.query_expander.expand_query(query)
                    expanded_queries.extend(expansions[:2])  # Limit expansions
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")

            # Retrieve relevant documents
            relevant_docs = await self._retrieve_documents(
                expanded_queries, k=kwargs.get("k", self.config.top_k * 2)
            )

            # Rerank if enabled
            if self.document_reranker and len(relevant_docs) > 1:
                try:
                    relevant_docs = self.document_reranker.rerank_documents(
                        query, relevant_docs, top_k=self.config.top_k
                    )
                except Exception as e:
                    logger.warning(f"Document reranking failed: {e}")

            # Generate response
            if not relevant_docs:
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing or provide more context."
                confidence = 0.1
                tokens_used = 0
            else:
                # Create context from documents
                context = self._create_context(relevant_docs[: self.config.top_k])

                # Generate prompt
                prompt = self._create_prompt(query, context, query_analysis)

                # Get response from Gemini
                gemini_response = await self.gemini_client.generate_response(
                    prompt,
                    temperature=kwargs.get("temperature", self.config.temperature),
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                )

                answer = gemini_response["text"]
                tokens_used = gemini_response["tokens_used"]
                confidence = self._calculate_confidence(
                    query_analysis, relevant_docs, answer
                )

            processing_time = time.time() - start_time

            # Create response
            response = RAGResponse(
                answer=answer,
                sources=relevant_docs[: self.config.top_k],
                query_analysis=query_analysis,
                confidence=confidence,
                processing_time=processing_time,
                tokens_used=tokens_used,
                cache_hit=cache_hit,
                error=error,
            )

            # Cache response
            if self.semantic_cache and not cache_hit and confidence > 0.5:
                self.semantic_cache.set(query, response)

            # Record metrics
            self.performance_monitor.record_query_metrics(
                query, processing_time, cache_hit, error is not None
            )

            return response

        except Exception as e:
            error = str(e)
            processing_time = time.time() - start_time

            logger.error(f"Error processing query: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Record error metrics
            self.performance_monitor.record_query_metrics(
                query, processing_time, cache_hit, True
            )

            return RAGResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {error}",
                sources=[],
                query_analysis=QueryAnalysis(
                    intent="error",
                    sentiment=0.0,
                    complexity=0.0,
                    topics=[],
                    entities=[],
                    question_type="error",
                    confidence=0.0,
                ),
                confidence=0.0,
                processing_time=processing_time,
                tokens_used=0,
                cache_hit=cache_hit,
                error=error,
            )

    async def _retrieve_documents(
        self, queries: List[str], k: int = 10
    ) -> List[Document]:
        """Retrieve documents for multiple queries"""
        all_documents = []
        seen_ids = set()

        for query in queries:
            try:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, self.vector_store.similarity_search, query, k
                )

                for doc in docs:
                    if (
                        doc.id not in seen_ids
                        and doc.score >= self.config.similarity_threshold
                    ):
                        all_documents.append(doc)
                        seen_ids.add(doc.id)

            except Exception as e:
                logger.warning(f"Error retrieving documents for query '{query}': {e}")

        # Sort by score and return top k
        all_documents.sort(key=lambda x: x.score or 0, reverse=True)
        return all_documents[:k]

    def _create_context(self, documents: List[Document]) -> str:
        """Create context string from documents"""
        if not documents:
            return "No relevant information found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Add document metadata for better context
            metadata = doc.metadata
            source_info = f"Source {i}"

            if metadata.get("title"):
                source_info += f" (Title: {metadata['title']})"
            if metadata.get("filename"):
                source_info += f" (File: {metadata['filename']})"

            context_parts.append(f"=== {source_info} ===")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line for separation

        return "\n".join(context_parts)

    def _create_prompt(
        self, query: str, context: str, query_analysis: QueryAnalysis
    ) -> str:
        """Create prompt for Gemini based on query analysis"""
        # Select system prompt based on query analysis
        system_prompt_key = self._select_system_prompt(query_analysis)
        system_prompt = self.system_prompts.get(
            system_prompt_key, self.system_prompts["default"]
        )

        # Create adaptive prompt based on query type
        prompt_parts = [
            system_prompt,
            "",
            "Context Information:",
            "=" * 50,
            context,
            "=" * 50,
            "",
            f"Query Analysis:",
            f"- Intent: {query_analysis.intent}",
            f"- Question Type: {query_analysis.question_type}",
            f"- Complexity: {query_analysis.complexity:.2f}",
            "",
        ]

        # Add specific instructions based on query type
        if query_analysis.question_type == "comparative":
            prompt_parts.append(
                "Please provide a detailed comparison highlighting key differences and similarities."
            )
        elif query_analysis.question_type == "procedural":
            prompt_parts.append(
                "Please provide step-by-step instructions or procedures."
            )
        elif query_analysis.question_type == "explanatory":
            prompt_parts.append(
                "Please provide a comprehensive explanation with reasoning."
            )
        elif query_analysis.question_type == "factual":
            prompt_parts.append("Please provide precise, factual information.")

        prompt_parts.extend(
            [
                "",
                f"User Question: {query}",
                "",
                "Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information, clearly state what information is missing. Always cite specific sources when possible.",
            ]
        )

        return "\n".join(prompt_parts)

    def _select_system_prompt(self, query_analysis: QueryAnalysis) -> str:
        """Select appropriate system prompt based on query analysis"""
        if query_analysis.intent == "procedural":
            return "technical"
        elif query_analysis.question_type == "comparative":
            return "analytical"
        elif "creative" in query_analysis.topics or "story" in query_analysis.topics:
            return "creative"
        else:
            return "default"

    def _calculate_confidence(
        self, query_analysis: QueryAnalysis, documents: List[Document], answer: str
    ) -> float:
        """Calculate confidence score for the response"""
        try:
            confidence_factors = []

            # Query analysis confidence
            confidence_factors.append(query_analysis.confidence)

            # Document relevance (average score)
            if documents:
                doc_scores = [doc.score for doc in documents if doc.score is not None]
                if doc_scores:
                    confidence_factors.append(sum(doc_scores) / len(doc_scores))
                else:
                    confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.1)

            # Answer quality indicators
            answer_length = len(answer.split())
            if 20 <= answer_length <= 200:
                confidence_factors.append(0.8)
            elif answer_length < 20:
                confidence_factors.append(0.4)
            else:
                confidence_factors.append(0.6)

            # Check for uncertainty indicators in answer
            uncertainty_phrases = [
                "i don't know",
                "not sure",
                "unclear",
                "insufficient information",
                "cannot determine",
                "unable to",
                "don't have enough",
            ]

            answer_lower = answer.lower()
            uncertainty_score = 1.0
            for phrase in uncertainty_phrases:
                if phrase in answer_lower:
                    uncertainty_score *= 0.5

            confidence_factors.append(uncertainty_score)

            # Calculate weighted average
            return min(1.0, sum(confidence_factors) / len(confidence_factors))

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Performance metrics
            perf_report = self.performance_monitor.get_performance_report()

            # Vector store stats
            vector_store_stats = {}
            if hasattr(self.vector_store, "get_statistics"):
                vector_store_stats = self.vector_store.get_statistics()

            # Document processing stats
            doc_metrics = self.document_processor.get_processing_metrics()

            # Cache statistics
            cache_stats = {}
            if self.semantic_cache:
                cache_stats = {
                    "cache_size": len(self.semantic_cache.cache),
                    "cache_capacity": self.semantic_cache.max_size,
                    "cache_utilization": len(self.semantic_cache.cache)
                    / self.semantic_cache.max_size,
                }

            return {
                "performance": perf_report,
                "vector_store": vector_store_stats,
                "document_processing": doc_metrics.__dict__,
                "cache": cache_stats,
                "configuration": {
                    "model_name": self.config.model_name,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "top_k": self.config.top_k,
                    "similarity_threshold": self.config.similarity_threshold,
                    "reranking_enabled": self.config.enable_reranking,
                    "query_expansion_enabled": self.config.enable_query_expansion,
                    "caching_enabled": self.config.enable_caching,
                },
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    async def clear_cache(self) -> bool:
        """Clear semantic cache"""
        try:
            if self.semantic_cache:
                self.semantic_cache.cache.clear()
                self.semantic_cache.embeddings.clear()
                self.semantic_cache.access_times.clear()
                logger.info("Semantic cache cleared")
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    async def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update RAG engine configuration"""
        try:
            # Update configuration attributes
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated config: {key} = {value}")

            # Reinitialize components if needed
            if "model_name" in new_config:
                self.gemini_client = GeminiClient(self.api_key, self.config.model_name)

            return True

        except Exception as e:
            logger.error(f"Error updating config: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            "overall": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Check Gemini API
            try:
                test_response = await self.gemini_client.generate_response(
                    "Hello, this is a health check.", max_tokens=10
                )
                health_status["components"]["gemini_api"] = {
                    "status": "healthy",
                    "response_time": "normal",
                }
            except Exception as e:
                health_status["components"]["gemini_api"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["overall"] = "degraded"

            # Check vector store
            try:
                if hasattr(self.vector_store, "similarity_search"):
                    test_docs = self.vector_store.similarity_search("test", k=1)
                    health_status["components"]["vector_store"] = {
                        "status": "healthy",
                        "document_count": len(test_docs)
                        if hasattr(self.vector_store, "__len__")
                        else "unknown",
                    }
                else:
                    health_status["components"]["vector_store"] = {
                        "status": "healthy",
                        "note": "basic functionality available",
                    }
            except Exception as e:
                health_status["components"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health_status["overall"] = "degraded"

            # Check ML components
            ml_components = [
                "query_analyzer",
                "query_expander",
                "document_reranker",
                "semantic_cache",
            ]
            for component in ml_components:
                if hasattr(self, component) and getattr(self, component):
                    health_status["components"][component] = {"status": "enabled"}
                else:
                    health_status["components"][component] = {"status": "disabled"}

            return health_status

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "overall": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Factory function
def create_rag_engine(api_key: str = None, **config_kwargs) -> RAGEngine:
    """Create RAG engine with configuration"""
    rag_config = RAGConfig(**config_kwargs)
    return RAGEngine(api_key=api_key, rag_config=rag_config)


# Export classes
__all__ = ["RAGEngine", "RAGResponse", "RAGConfig", "GeminiClient", "create_rag_engine"]
