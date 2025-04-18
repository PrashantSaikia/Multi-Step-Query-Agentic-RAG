import logging
from pathlib import Path
import argparse
from typing import Optional

from config import VECTOR_STORE_DIR
from agents import RAGAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main(question: Optional[str] = None) -> None:
    """Main function to run the RAG application."""
    try:
        # Check if vector store exists
        if not VECTOR_STORE_DIR.exists() or not list(VECTOR_STORE_DIR.glob("*")):
            logger.error("Vector store not found. Please run ingest_docs.py first to process documents.")
            return
        
        # Initialize RAG agent
        agent = RAGAgent()
        
        if question:
            # Process the question
            response = agent.process_question(question)
            print("\nResponse:")
            print(response)
        else:
            # Interactive mode
            print("Enter your questions (type 'exit' to quit):")
            while True:
                question = input("\nQuestion: ").strip()
                if question.lower() == 'exit':
                    break
                
                response = agent.process_question(question)
                print("\nResponse:")
                print(response)
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-step RAG application")
    parser.add_argument("--question", type=str, help="Question to process (optional)")
    args = parser.parse_args()
    
    main(args.question) 