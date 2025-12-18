"""
Main entry point for the Advanced RAG Chatbot
"""
import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TF optimizations

# Load environment variables
load_dotenv()

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_chatbot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_environment():
    """Set up the environment and check dependencies"""
    logger.info("Setting up environment...")
    
    # Create necessary directories
    directories = [
        './data',
        './data/chromadb',
        './data/faiss_index',
        './models',
        './temp_uploads',
        './logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")
    
    # Check for required environment variables
    required_env_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Please set these variables in your .env file or environment")
    
    logger.info("Environment setup complete")

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'google-generativeai',
        'sentence-transformers',
        'chromadb',
        'faiss-cpu',
        'scikit-learn',
        'pandas',
        'numpy',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed")
    return True

async def test_rag_system():
    """Test the RAG system components"""
    logger.info("Testing RAG system components...")
    
    try:
        # Test imports
        from config import config
        from rag_engine import RAGEngine
        from vector_store import create_vector_store
        from document_processor import DocumentProcessor
        
        logger.info("✅ All imports successful")
        
        # Test configuration
        if not config.validate_config():
            logger.error("❌ Configuration validation failed")
            return False
        
        logger.info("✅ Configuration validation passed")
        
        # Test document processor
        doc_processor = DocumentProcessor()
        logger.info("✅ Document processor initialized")
        
        # Test vector store
        vector_store = create_vector_store()
        logger.info("✅ Vector store initialized")
        
        # Test RAG engine (requires API key)
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            try:
                rag_engine = RAGEngine(api_key=api_key)
                
                # Test health check
                health_status = await rag_engine.health_check()
                if health_status.get('overall') in ['healthy', 'degraded']:
                    logger.info("✅ RAG engine health check passed")
                else:
                    logger.warning("⚠️ RAG engine health check showed issues")
                
            except Exception as e:
                logger.error(f"❌ RAG engine test failed: {e}")
                return False
        else:
            logger.warning("⚠️ No Gemini API key found, skipping RAG engine test")
        
        logger.info("🎉 All system tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit application"""
    logger.info("Starting Streamlit application...")
    
    # Set Streamlit configuration
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Import and run Streamlit
    try:
        import streamlit.web.cli as stcli
        sys.argv = ['streamlit', 'run', 'streamlit_app.py']
        sys.exit(stcli.main())
    except Exception as e:
        logger.error(f"Failed to start Streamlit app: {e}")
        sys.exit(1)

def run_cli_mode():
    """Run in CLI mode for testing and development"""
    logger.info("Starting CLI mode...")
    
    async def cli_session():
        from rag_engine import RAGEngine
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is required")
            return
        
        try:
            # Initialize RAG engine
            rag_engine = RAGEngine(api_key=api_key)
            logger.info("RAG engine initialized successfully")
            
            # Interactive loop
            print("\n🤖 Advanced RAG Chatbot - CLI Mode")
            print("Type 'quit' to exit, 'help' for commands\n")
            
            while True:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                    
                    if user_input.lower() == 'help':
                        print("Available commands:")
                        print("  quit/exit/q - Exit the application")
                        print("  stats - Show system statistics")
                        print("  health - Run health check")
                        print("  clear - Clear cache")
                        continue
                    
                    if user_input.lower() == 'stats':
                        stats = await rag_engine.get_statistics()
                        print(f"\nSystem Statistics:")
                        for key, value in stats.items():
                            print(f"  {key}: {value}")
                        continue
                    
                    if user_input.lower() == 'health':
                        health = await rag_engine.health_check()
                        print(f"\nHealth Status: {health.get('overall', 'unknown')}")
                        continue
                    
                    if user_input.lower() == 'clear':
                        await rag_engine.clear_cache()
                        print("Cache cleared!")
                        continue
                    
                    if not user_input:
                        continue
                    
                    # Process query
                    print("Assistant: Thinking...")
                    response = await rag_engine.query(user_input)
                    
                    print(f"Assistant: {response.answer}")
                    print(f"(Confidence: {response.confidence:.2%}, "
                          f"Time: {response.processing_time:.2f}s, "
                          f"Sources: {len(response.sources)})")
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in CLI mode: {e}")
                    print(f"Error: {e}")
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
    
    asyncio.run(cli_session())

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Advanced RAG Chatbot')
    parser.add_argument('--mode', choices=['web', 'cli', 'test'], default='web',
                       help='Run mode: web (Streamlit), cli (command line), or test')
    parser.add_argument('--setup', action='store_true',
                       help='Run initial setup only')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Always setup environment
    setup_environment()
    
    # Check dependencies
    if args.check_deps or args.mode == 'test':
        if not check_dependencies():
            sys.exit(1)
        
        if args.check_deps:
            logger.info("Dependency check complete")
            return
    
    # Run setup only
    if args.setup:
        logger.info("Setup complete")
        return
    
    # Run in specified mode
    if args.mode == 'test':
        logger.info("Running system tests...")
        success = asyncio.run(test_rag_system())
        sys.exit(0 if success else 1)
    
    elif args.mode == 'cli':
        run_cli_mode()
    
    elif args.mode == 'web':
        run_streamlit_app()
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    if 'streamlit' in sys.modules:
        import streamlit_app
    else:
        main()