from pathlib import Path
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
import logging

from config import VECTOR_STORE_DIR, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
        self.vector_store = None
        self.index_path = VECTOR_STORE_DIR / "faiss_index"

    def create_vector_store(self, chunks: List[Dict]):
        """Create and save a vector store from document chunks."""
        try:
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # Save the vector store
            self.vector_store.save_local(str(self.index_path))
            logger.info("Vector store created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def load_vector_store(self):
        """Load an existing vector store."""
        try:
            if not self.index_path.exists():
                raise ValueError("Vector store not found. Please run ingest_docs.py first.")
            
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search the vector store for relevant chunks."""
        if not self.vector_store:
            self.load_vector_store()
        
        try:
            logger.info(f"Performing semantic search for query: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise 