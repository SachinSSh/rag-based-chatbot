"""
Machine Learning utilities for enhanced RAG performance
"""
import os
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import defaultdict, deque
import threading
import sqlite3
import json

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import spacy

# Deep learning imports
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer, pipeline
)
import torch

from config import config, ML_MODEL_PATHS, PERFORMANCE_METRICS

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

@dataclass
class QueryAnalysis:
    """Query analysis results"""
    intent: str
    sentiment: float
    complexity: float
    topics: List[str]
    entities: List[str]
    question_type: str
    confidence: float

@dataclass
class DocumentQuality:
    """Document quality metrics"""
    relevance_score: float
    readability_score: float
    completeness_score: float
    freshness_score: float
    authority_score: float
    overall_score: float

class SemanticCache:
    """Advanced semantic caching system"""
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.embeddings = {}
        self.lock = threading.Lock()
        
        # Initialize embedder for semantic similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # SQLite for persistent cache
        self.db_path = "./data/semantic_cache.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database for persistent cache"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    embedding BLOB,
                    timestamp REAL,
                    access_count INTEGER DEFAULT 1
                )
            ''')
            conn.commit()
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, query: str, similarity_threshold: float = 0.85) -> Any:
        """Get cached result with semantic similarity"""
        with self.lock:
            query_embedding = self.embedder.encode(query)
            
            # Check exact match first
            exact_key = self._get_cache_key(query)
            if exact_key in self.cache:
                if time.time() - self.access_times[exact_key] < self.ttl:
                    return self.cache[exact_key]
                else:
                    self._remove_key(exact_key)
            
            # Check semantic similarity
            best_match = None
            best_similarity = 0.0
            
            for key, embedding in self.embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = key
            
            if best_match and time.time() - self.access_times[best_match] < self.ttl:
                return self.cache[best_match]
            
            return None
    
    def set(self, query: str, value: Any) -> None:
        """Set cached value"""
        with self.lock:
            key = self._get_cache_key(query)
            embedding = self.embedder.encode(query)
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = value
            self.embeddings[key] = embedding
            self.access_times[key] = time.time()
            
            # Save to persistent storage
            self._save_to_db(key, value, embedding)
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entries"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache"""
        self.cache.pop(key, None)
        self.embeddings.pop(key, None)
        self.access_times.pop(key, None)
    
    def _save_to_db(self, key: str, value: Any, embedding: np.ndarray) -> None:
        """Save to SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, embedding, timestamp) 
                    VALUES (?, ?, ?, ?)
                ''', (
                    key, 
                    pickle.dumps(value), 
                    pickle.dumps(embedding),
                    time.time()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving to cache DB: {e}")

class QueryExpander:
    """Intelligent query expansion using multiple techniques"""
    
    def __init__(self):
        # Load models
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # Initialize TF-IDF for term expansion
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
        self.is_fitted = False
        
        # Synonym expansion
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query using multiple techniques"""
        expansions = []
        
        try:
            # 1. Synonym expansion
            synonym_expansion = self._expand_with_synonyms(query)
            if synonym_expansion != query:
                expansions.append(synonym_expansion)
            
            # 2. Paraphrase generation
            paraphrases = self._generate_paraphrases(query, max_expansions)
            expansions.extend(paraphrases)
            
            # 3. Entity-based expansion
            if self.nlp:
                entity_expansion = self._expand_with_entities(query)
                if entity_expansion != query:
                    expansions.append(entity_expansion)
            
            # 4. Contextual expansion
            contextual_expansion = self._contextual_expansion(query)
            expansions.extend(contextual_expansion)
            
        except Exception as e:
            logger.error(f"Error in query expansion: {e}")
        
        return list(set(expansions))[:max_expansions]
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms using TextBlob"""
        blob = TextBlob(query)
        expanded_words = []
        
        for word in blob.words:
            synsets = word.synsets
            if synsets:
                # Get most common synonym
                synonyms = [lemma.name() for synset in synsets[:1] for lemma in synset.lemmas()[:2]]
                if synonyms and synonyms[0] != word:
                    expanded_words.append(f"({word} OR {synonyms[0]})")
                else:
                    expanded_words.append(str(word))
            else:
                expanded_words.append(str(word))
        
        return " ".join(expanded_words)
    
    def _generate_paraphrases(self, query: str, num_paraphrases: int = 2) -> List[str]:
        """Generate paraphrases using T5"""
        try:
            input_text = f"paraphrase: {query} </s>"
            inputs = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            paraphrases = []
            for _ in range(num_paraphrases):
                outputs = self.t5_model.generate(
                    inputs, 
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
                
                paraphrase = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if paraphrase != query and paraphrase not in paraphrases:
                    paraphrases.append(paraphrase)
            
            return paraphrases
        
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
            return []
    
    def _expand_with_entities(self, query: str) -> str:
        """Expand query by identifying and enhancing entities"""
        doc = self.nlp(query)
        expanded_query = query
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT"]:
                # Add entity type information
                expanded_query = expanded_query.replace(
                    ent.text, 
                    f"{ent.text} ({ent.label_.lower()})"
                )
        
        return expanded_query
    
    def _contextual_expansion(self, query: str) -> List[str]:
        """Generate contextual expansions"""
        expansions = []
        
        # Question type specific expansions
        if query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            # Add alternative question formulations
            if query.lower().startswith('what'):
                expansions.append(query.replace('What', 'Which', 1))
                expansions.append(query.replace('what', 'define', 1))
            elif query.lower().startswith('how'):
                expansions.append(query.replace('How', 'What is the method', 1))
                expansions.append(query.replace('how', 'steps to', 1))
        
        return expansions

class DocumentReranker:
    """Advanced document reranking using cross-encoders and ML models"""
    
    def __init__(self):
        # Load cross-encoder for reranking
        self.cross_encoder = CrossEncoder(config.ml.reranker_model)
        
        # Initialize feature extractors
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # ML models for ranking
        self.ranking_model = None
        self.feature_columns = []
        
        # Load pre-trained model if available
        self._load_ranking_model()
    
    def rerank_documents(self, query: str, documents: List, top_k: int = 10) -> List:
        """Rerank documents using multiple signals"""
        if not documents:
            return documents
        
        try:
            # 1. Cross-encoder reranking
            cross_encoder_scores = self._cross_encoder_rerank(query, documents)
            
            # 2. Feature-based reranking
            feature_scores = self._feature_based_rerank(query, documents)
            
            # 3. Combine scores
            final_scores = []
            for i, doc in enumerate(documents):
                combined_score = (
                    0.6 * cross_encoder_scores[i] + 
                    0.4 * feature_scores[i]
                )
                final_scores.append((combined_score, doc))
            
            # Sort by combined score
            final_scores.sort(key=lambda x: x[0], reverse=True)
            
            return [doc for _, doc in final_scores[:top_k]]
        
        except Exception as e:
            logger.error(f"Error in document reranking: {e}")
            return documents[:top_k]
    
    def _cross_encoder_rerank(self, query: str, documents: List) -> List[float]:
        """Rerank using cross-encoder"""
        pairs = [[query, doc.content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else scores
    
    def _feature_based_rerank(self, query: str, documents: List) -> List[float]:
        """Rerank using engineered features"""
        features = []
        
        for doc in documents:
            doc_features = self._extract_features(query, doc)
            features.append(doc_features)
        
        if self.ranking_model and features:
            try:
                features_df = pd.DataFrame(features)
                # Ensure all required columns are present
                for col in self.feature_columns:
                    if col not in features_df.columns:
                        features_df[col] = 0.0
                
                scores = self.ranking_model.predict(features_df[self.feature_columns])
                return scores.tolist()
            except Exception as e:
                logger.error(f"Error in feature-based ranking: {e}")
        
        # Fallback to simple scoring
        return [self._simple_score(query, doc) for doc in documents]
    
    def _extract_features(self, query: str, document) -> Dict[str, float]:
        """Extract ranking features"""
        doc_content = getattr(document, 'content', str(document))
        doc_metadata = getattr(document, 'metadata', {})
        
        features = {
            # Text similarity features
            'query_doc_similarity': self._calculate_similarity(query, doc_content),
            'query_length': len(query.split()),
            'doc_length': len(doc_content.split()),
            'length_ratio': len(query.split()) / max(len(doc_content.split()), 1),
            
            # Content quality features
            'readability': self._calculate_readability(doc_content),
            'information_density': self._calculate_information_density(doc_content),
            'keyword_coverage': self._calculate_keyword_coverage(query, doc_content),
            
            # Metadata features
            'has_title': 1.0 if doc_metadata.get('title') else 0.0,
            'has_author': 1.0 if doc_metadata.get('author') else 0.0,
            'has_date': 1.0 if doc_metadata.get('date') else 0.0,
            'source_quality': self._estimate_source_quality(doc_metadata),
            
            # Position features (if available)
            'original_rank': getattr(document, 'original_rank', 0),
            'retrieval_score': getattr(document, 'score', 0.0),
        }
        
        return features
    
    def _calculate_similarity(self, query: str, doc: str) -> float:
        """Calculate text similarity"""
        try:
            # Simple TF-IDF based similarity
            texts = [query, doc]
            tfidf_matrix = self.tfidf.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            avg_sentence_length = len(words) / len(sentences)
            # Simplified readability score
            score = max(0, min(100, 206.835 - 1.015 * avg_sentence_length))
            return score / 100.0
        except:
            return 0.5
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density"""
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            content_words = [w for w in words if w not in stop_words and w.isalpha()]
            
            if len(words) == 0:
                return 0.0
            
            return len(set(content_words)) / len(words)
        except:
            return 0.0
    
    def _calculate_keyword_coverage(self, query: str, doc: str) -> float:
        """Calculate query keyword coverage in document"""
        try:
            query_words = set(word_tokenize(query.lower()))
            doc_words = set(word_tokenize(doc.lower()))
            
            if len(query_words) == 0:
                return 0.0
            
            covered_words = query_words.intersection(doc_words)
            return len(covered_words) / len(query_words)
        except:
            return 0.0
    
    def _estimate_source_quality(self, metadata: Dict) -> float:
        """Estimate source quality based on metadata"""
        score = 0.5  # Base score
        
        # Boost for academic sources
        source = metadata.get('source', '').lower()
        if any(term in source for term in ['arxiv', 'pubmed', 'ieee', 'acm']):
            score += 0.3
        elif any(term in source for term in ['wikipedia', 'gov', 'edu']):
            score += 0.2
        
        # Boost for recent content
        if 'date' in metadata:
            # Simplified recency boost
            score += 0.1
        
        return min(1.0, score)
    
    def _simple_score(self, query: str, document) -> float:
        """Simple fallback scoring"""
        doc_content = getattr(document, 'content', str(document))
        similarity = self._calculate_similarity(query, doc_content)
        length_penalty = 1.0 / (1.0 + len(doc_content) / 10000)  # Prefer shorter docs
        return similarity * length_penalty
    
    def _load_ranking_model(self) -> None:
        """Load pre-trained ranking model"""
        model_path = ML_MODEL_PATHS.get('quality_scorer')
        if model_path and os.path.exists(f"{model_path}.pkl"):
            try:
                with open(f"{model_path}.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                    self.ranking_model = model_data['model']
                    self.feature_columns = model_data['features']
                logger.info("Loaded pre-trained ranking model")
            except Exception as e:
                logger.error(f"Error loading ranking model: {e}")

class QueryAnalyzer:
    """Advanced query analysis and classification"""
    
    def __init__(self):
        # Load classification models
        self.intent_classifier = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            return_all_scores=True
        ) if torch.cuda.is_available() else None
        
        # Initialize components
        self.sentiment_analyzer = TextBlob
        self.complexity_analyzer = self._init_complexity_analyzer()
        
        # Topic modeling
        self.topic_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Load spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        try:
            # Basic analysis
            sentiment = self._analyze_sentiment(query)
            complexity = self._analyze_complexity(query)
            
            # Intent classification
            intent = self._classify_intent(query)
            
            # Topic extraction
            topics = self._extract_topics(query)
            
            # Entity extraction
            entities = self._extract_entities(query)
            
            # Question type classification
            question_type = self._classify_question_type(query)
            
            # Confidence estimation
            confidence = self._estimate_confidence(query)
            
            return QueryAnalysis(
                intent=intent,
                sentiment=sentiment,
                complexity=complexity,
                topics=topics,
                entities=entities,
                question_type=question_type,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return QueryAnalysis(
                intent="unknown",
                sentiment=0.0,
                complexity=0.5,
                topics=[],
                entities=[],
                question_type="general",
                confidence=0.5
            )
    
    def _analyze_sentiment(self, query: str) -> float:
        """Analyze query sentiment"""
        try:
            blob = TextBlob(query)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def _analyze_complexity(self, query: str) -> float:
        """Analyze query complexity"""
        try:
            # Simple complexity metrics
            word_count = len(query.split())
            char_count = len(query)
            question_marks = query.count('?')
            
            # Normalize to 0-1 scale
            complexity = min(1.0, (word_count * 0.05 + char_count * 0.001 + question_marks * 0.1))
            return complexity
        except:
            return 0.5
    
    def _classify_intent(self, query: str) -> str:
        """Classify query intent"""
        query_lower = query.lower().strip()
        
        # Simple rule-based intent classification
        if query_lower.startswith(('what', 'define', 'explain')):
            return "information_seeking"
        elif query_lower.startswith(('how', 'show me')):
            return "procedural"
        elif query_lower.startswith(('why', 'because')):
            return "explanatory"
        elif query_lower.startswith(('when', 'where')):
            return "factual"
        elif query_lower.startswith(('compare', 'difference')):
            return "comparative"
        elif '?' in query:
            return "question"
        else:
            return "general"
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from query"""
        try:
            # Simple keyword extraction
            words = word_tokenize(query.lower())
            stop_words = set(stopwords.words('english'))
            keywords = [word for word in words if word.isalpha() and word not in stop_words]
            
            # Return top keywords as topics
            return keywords[:5]
        except:
            return []
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(query)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"]]
            return entities
        except:
            return []
    
    def _classify_question_type(self, query: str) -> str:
        """Classify question type"""
        query_lower = query.lower().strip()
        
        question_types = {
            'factual': ['what is', 'what are', 'who is', 'who are', 'when', 'where'],
            'procedural': ['how to', 'how do', 'how can', 'steps to'],
            'explanatory': ['why', 'because', 'explain', 'reason'],
            'comparative': ['compare', 'difference', 'versus', 'vs', 'better'],
            'yes_no': ['is', 'are', 'can', 'will', 'should', 'does', 'do'],
        }
        
        for q_type, patterns in question_types.items():
            if any(pattern in query_lower for pattern in patterns):
                return q_type
        
        return 'general'
    
    def _estimate_confidence(self, query: str) -> float:
        """Estimate confidence in understanding the query"""
        try:
            factors = []
            
            # Length factor
            word_count = len(query.split())
            if 3 <= word_count <= 20:
                factors.append(0.8)
            else:
                factors.append(0.6)
            
            # Grammar factor (simplified)
            if query.strip().endswith(('?', '.', '!')):
                factors.append(0.9)
            else:
                factors.append(0.7)
            
            # Spelling factor (simplified check)
            blob = TextBlob(query)
            if len(str(blob.correct())) == len(query):
                factors.append(0.9)
            else:
                factors.append(0.6)
            
            return sum(factors) / len(factors)
        except:
            return 0.5
    
    def _init_complexity_analyzer(self):
        """Initialize complexity analyzer"""
        # This would typically load a pre-trained model
        return None

class PerformanceMonitor:
    """Monitor and optimize RAG system performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=100)
        self.start_time = time.time()
        
        # Performance thresholds
        self.thresholds = {
            'response_time': config.ml.response_time_threshold,
            'memory_usage': config.ml.memory_threshold,
            'error_rate': 0.05,
            'cache_hit_rate': 0.6
        }
    
    def record_query_metrics(self, query: str, response_time: float, cache_hit: bool, error: bool = False):
        """Record metrics for a query"""
        self.metrics['response_times'].append(response_time)
        self.metrics['cache_hits'].append(1 if cache_hit else 0)
        self.metrics['errors'].append(1 if error else 0)
        self.metrics['queries'].append(time.time())
        
        # Update performance metrics
        PERFORMANCE_METRICS["response_time"].append(response_time)
        PERFORMANCE_METRICS["query_count"] += 1
        if error:
            PERFORMANCE_METRICS["error_count"] += 1
        
        # Check for alerts
        self._check_alerts()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics['response_times']:
            return {"status": "No data available"}
        
        response_times = self.metrics['response_times']
        cache_hits = self.metrics['cache_hits']
        errors = self.metrics['errors']
        
        # Calculate cache hit rate
        cache_hit_rate = sum(cache_hits) / len(cache_hits) if cache_hits else 0
        error_rate = sum(errors) / len(errors) if errors else 0
        
        report = {
            'total_queries': len(response_times),
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'queries_per_hour': len(response_times) / max(1, (time.time() - self.start_time) / 3600)
        }
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _check_alerts(self):
        """Check for performance alerts"""
        if len(self.metrics['response_times']) < 10:
            return
        
        recent_response_times = self.metrics['response_times'][-10:]
        avg_response_time = np.mean(recent_response_times)
        
        if avg_response_time > self.thresholds['response_time']:
            alert = {
                'type': 'high_response_time',
                'value': avg_response_time,
                'threshold': self.thresholds['response_time'],
                'timestamp': time.time()
            }
            self.alerts.append(alert)
            logger.warning(f"High response time alert: {avg_response_time:.2f}s")
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if report['avg_response_time'] > self.thresholds['response_time']:
            recommendations.append("Consider reducing document chunk size or increasing cache size")
        
        if report['cache_hit_rate'] < self.thresholds['cache_hit_rate']:
            recommendations.append("Increase semantic cache size or improve cache TTL settings")
        
        if report['error_rate'] > self.thresholds['error_rate']:
            recommendations.append("Review error logs and improve input validation")
        
        if report['p99_response_time'] > report['avg_response_time'] * 3:
            recommendations.append("Investigate query complexity variations and optimize slow queries")
        
        return recommendations

# Export classes
__all__ = [
    'SemanticCache',
    'QueryExpander', 
    'DocumentReranker',
    'QueryAnalyzer',
    'PerformanceMonitor',
    'QueryAnalysis',
    'DocumentQuality'
]