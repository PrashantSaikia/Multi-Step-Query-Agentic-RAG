import logging
from pathlib import Path
import shutil
import time

from config import DOCS_DIR, VECTOR_STORE_DIR
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_vector_store() -> None:
    """Clean up the vector store directory."""
    try:
        # Delete the vector store directory
        if VECTOR_STORE_DIR.exists():
            logger.info("Deleting existing vector store...")
            shutil.rmtree(VECTOR_STORE_DIR)
            time.sleep(1)  # Give the system time to complete the deletion
        
        # Recreate the directory
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
        
    except Exception as e:
        logger.error(f"Error cleaning vector store: {e}")
        raise

def ingest_documents() -> None:
    """Process documents and create vector store."""
    try:
        # Check if Docs directory exists and has PDFs
        if not DOCS_DIR.exists():
            logger.error(f"Docs directory not found at {DOCS_DIR}")
            return
        
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {DOCS_DIR}")
            return

        # Clean up existing vector store
        clean_vector_store()

        # Process documents and create vector store
        logger.info("Processing documents...")
        processor = DocumentProcessor()
        chunks = processor.process_documents()
        
        logger.info("Creating vector store...")
        vector_store = VectorStoreManager()
        vector_store.create_vector_store(chunks)
        logger.info("Vector store created successfully")
    
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise

if __name__ == "__main__":
    ingest_documents() 