# 🤖 Advanced RAG Chatbot with Gemini API

A comprehensive Retrieval-Augmented Generation (RAG) chatbot built with Google's Gemini API, featuring advanced ML algorithms, semantic caching, document reranking, and a beautiful Streamlit interface.

## 🚀 Features

### Core RAG Capabilities
- **Multi-format Document Support**: PDF, DOCX, TXT, MD, HTML, CSV, JSON, XML
- **Intelligent Document Processing**: Semantic chunking, metadata extraction, quality scoring
- **Advanced Vector Storage**: ChromaDB, FAISS, and hybrid storage options
- **Semantic Search**: High-quality embeddings with similarity search

### ML Enhancements
- **Query Analysis & Classification**: Intent detection, sentiment analysis, complexity scoring
- **Query Expansion**: Automatic query enhancement using T5 and synonym expansion
- **Document Reranking**: Cross-encoder models for improved relevance
- **Semantic Caching**: Intelligent caching based on semantic similarity
- **Performance Monitoring**: Real-time metrics and optimization recommendations

### Advanced Features
- **Multi-model Support**: Gemini Pro, with extensibility for other LLMs
- **Streaming Responses**: Real-time response generation
- **Error Handling**: Comprehensive error handling and recovery
- **Health Monitoring**: System health checks and diagnostics
- **Analytics Dashboard**: Performance visualization and insights

### User Interface
- **Modern Streamlit UI**: Clean, responsive design with dark/light themes
- **Chat Interface**: Interactive chat with message history
- **Document Management**: Easy file upload and processing
- **Real-time Analytics**: Performance metrics and visualizations
- **System Health Dashboard**: Component status and diagnostics

## 📁 Project Structure

```
advanced-rag-chatbot/
├── requirements.txt           # Python dependencies
├── config.py                 # Configuration management
├── vector_store.py           # Vector database implementations
├── ml_utils.py              # ML algorithms and utilities
├── document_processor.py    # Document processing pipeline
├── rag_engine.py            # Main RAG engine
├── streamlit_app.py         # Streamlit web interface
├── main.py                  # Entry point and CLI
├── .env.example             # Environment variables template
├── README.md                # This file
├── data/                    # Data storage
│   ├── chromadb/           # ChromaDB persistence
│   ├── faiss_index/        # FAISS indices
│   └── semantic_cache.db   # SQLite cache
├── models/                  # ML model storage
├── logs/                   # Application logs
└── temp_uploads/           # Temporary file storage
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+ 
- Google Gemini API key
- 8GB+ RAM recommended
- 2GB+ disk space

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/advanced-rag-chatbot.git
cd advanced-rag-chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLP Models
```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Set Up Environment Variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Step 6: Initialize the System
```bash
python main.py --setup
```

## 🔑 Configuration

### Required API Keys
1. **Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Optional Keys**: OpenAI, Cohere, Anthropic, HuggingFace for extended functionality

### Environment Variables
Edit the `.env` file with your configuration:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional for enhanced features
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_hf_token

# Performance tuning
CHUNK_SIZE=512
SIMILARITY_THRESHOLD=0.7
ENABLE_RERANKING=true
ENABLE_CACHING=true
```

## 🚀 Usage

### Web Interface (Recommended)
```bash
python main.py --mode web
```
Then open http://localhost:8501 in your browser.

### Command Line Interface
```bash
python main.py --mode cli
```

### System Testing
```bash
python main.py --mode test
```

### Direct Streamlit
```bash
streamlit run streamlit_app.py
```

## 💻 Using the Application

### 1. Initialize the System
- Enter your Gemini API key in the sidebar
- Configure model parameters (temperature, max tokens, etc.)
- Enable desired ML features
- Click "Initialize RAG Engine"

### 2. Upload Documents
- Go to the "Documents" tab
- Upload files (PDF, DOCX, TXT, etc.)
- Click "Process and Add Documents"
- Wait for processing to complete

### 3. Start Chatting
- Go to the "Chat" tab
- Ask questions about your documents
- View sources and confidence scores
- Explore response metadata

### 4. Monitor Performance
- Check the "Analytics" tab for performance metrics
- View response time trends and confidence distributions
- Monitor system health in the "System Health" tab

## 🧠 ML Algorithms & Components

### Document Processing
- **Semantic Chunking**: Context-aware text splitting
- **Quality Scoring**: Multi-factor document quality assessment
- **Metadata Extraction**: Automatic title, topic, and entity extraction
- **Language Detection**: Automatic language identification

### Query Processing
- **Intent Classification**: Categorizes user queries by intent
- **Query Expansion**: Enhances queries with synonyms and paraphrases
- **Complexity Analysis**: Measures query complexity and difficulty
- **Entity Recognition**: Extracts named entities from queries

### Retrieval Enhancement
- **Hybrid Search**: Combines multiple retrieval strategies
- **Document Reranking**: Cross-encoder models for relevance scoring
- **Semantic Caching**: Caches responses for similar queries
- **Adaptive Retrieval**: Adjusts retrieval based on query type

### Performance Optimization
- **Response Time Monitoring**: Tracks and optimizes response times
- **Memory Management**: Efficient memory usage and garbage collection
- **Caching Strategies**: Multi-level caching for improved performance
- **Load Balancing**: Distributes processing across resources

## 📊 Analytics & Monitoring

### Performance Metrics
- Response time statistics (average, P95, P99)
- Confidence score distributions
- Cache hit rates and efficiency
- Error rates and failure analysis

### Query Analytics
- Intent and question type distributions
- Complexity analysis over time
- Popular topics and entities
- User interaction patterns

### System Health
- Component status monitoring
- Resource usage tracking
- Error logging and alerting
- Automated performance recommendations

## 🔧 Advanced Configuration

### Vector Store Options
```python
# ChromaDB (default)
vector_store_type = "chromadb"

# FAISS for high-performance
vector_store_type = "faiss"

# Hybrid for best of both
vector_store_type = "hybrid"
```

### ML Model Customization
```python
# Custom embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Custom reranking model
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Enable/disable features
enable_reranking = True
enable_query_expansion = True
enable_semantic_caching = True
```

### Performance Tuning
```python
# Chunk size optimization
chunk_size = 512
chunk_overlap = 50

# Retrieval parameters
similarity_threshold = 0.7
max_retrieval_docs = 10

# Response generation
temperature = 0.7
max_tokens = 2048
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. API Key Issues**
- Verify your Gemini API key is correct
- Check API quotas and billing
- Ensure API key has proper permissions

**3. Memory Issues**
- Reduce chunk size in config.py
- Enable garbage collection
- Use FAISS instead of ChromaDB for large datasets

**4. Slow Performance**
- Enable semantic caching
- Reduce similarity threshold
- Use smaller embedding models

### Error Logs
Check `rag_chatbot.log` for detailed error information:
```bash
tail -f rag_chatbot.log
```

### Health Check
```bash
python main.py --mode test
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Integration Tests
```bash
python main.py --mode test
```

### Performance Testing
```bash
python -m locust -f tests/load_test.py
```

## 🚀 Deployment

### Local Development
```bash
python main.py --mode web
```

### Docker Deployment
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key rag-chatbot
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub
- **AWS/GCP/Azure**: Use containerization
- **Kubernetes**: Provided YAML configurations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black .
isort .

# Run linting
flake8 .

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google for the Gemini API
- Streamlit for the amazing web framework
- ChromaDB and FAISS for vector storage
- Sentence Transformers for embeddings
- The open-source ML community

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/advanced-rag-chatbot/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/advanced-rag-chatbot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/advanced-rag-chatbot/discussions)
- **Email**: support@yourproject.com

## 🗺️ Roadmap

### Version 2.1 (Next Release)
- [ ] Multi-modal support (images, audio)
- [ ] Advanced conversation memory
- [ ] Custom model fine-tuning
- [ ] Enhanced security features

### Version 2.2 (Future)
- [ ] Collaborative chat features
- [ ] Advanced analytics dashboard
- [ ] Mobile application
- [ ] Enterprise SSO integration

### Version 3.0 (Long-term)
- [ ] Distributed processing
- [ ] Real-time learning
- [ ] Advanced reasoning capabilities
- [ ] Multi-language support

---

**Built with ❤️ by the Advanced RAG Team**

*Star ⭐ this repository if you find it helpful!*
python main.py --mode web
# Or use the deployment script
./deploy.sh local
./deploy.sh docker