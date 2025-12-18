import asyncio
import os
import sys
from pathlib import Path

import rag_engine

# adding parent directory
sys.path.insert(0, str(Path(__file__).parent))

from rag_engine import RAGEngine
from dotenv import load_dotenv

load_dotenv()


async def test_vector_store():
    "Test vector functionality"
    print("=" * 60)
    print("RAG System Diagnostic Test")
    print("=" * 60)

    # check api key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in ur environments!")
        return
    print("!! GEMINI KEY FOUND")

    # initilize RAG engine
    print("\n 1. Initializing RAG engine>>>")
    try:
        rag_engine = RAGEngine(api_key=api_key)
        print("RAG engine successfully started")
    except Exception as e:
        print(f" Failed to initialize RAG engine: {e}")
        return

    # check vector store
    print("\n 2. Checking vector store>>>")
    if rag_engine.vector_store is None:
        print("VEctor store is None!")
        return

    if hasattr(rag_engine.vector_store, "collection"):
        doc_count = rag_engine.vector_store.collection.count()
        print(f"Vector store initialized: {type(rag_engine.vector_store).__name__}")
        print(f"Documnet count: {doc_count}")

        if doc_count == 0:
            print("\n warning!!!: vEctor store is empty")
            print(" u need to uplaod documnets via the streamlit app first ")
        else:
            print(f"\n vector store contains {doc_count} document chucks")

            # test a simple query
            test_query = "test query"

            try:
                response = await rag_engine.query(test_query)
                print(f"\n Query results:")
                print(f" Answer: {response.answer[:200]}...")
                print(f" Confidence: {response.confidence:.2%}")
                print(f" Sources found: {len(response.sources)}")
                print(f" Processing time: {response.processing_time:.2f}s")

                if len(response.sources) > 0:
                    print("\n document retrieval is Working !!")
                    for i, source in enumerate(response.sources[:3], 1):
                        print(i)
                        print(source.score)
                        print(source.content)
                else:
                    print("\n No document retrieval")

            except Exception as e:
                print(f" \n Query failed: {e}")
                import traceback

                traceback.print_exc()
    else:
        print("vector store doesn't have any collection attributes")

    print("\n" + "==" * 30)
    print("Diagnostic eneded!")


if __name__ == "__main__":
    asyncio.run(test_vector_store())
