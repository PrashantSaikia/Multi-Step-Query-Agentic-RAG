from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pypdf import PdfReader
import logging

from config import DOCS_DIR

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def convert_pdf_to_markdown(self, pdf_path: Path) -> str:
        """Convert a PDF file to markdown using pypdf."""
        try:
            reader = PdfReader(str(pdf_path))
            markdown_content = []
            
            for page in reader.pages:
                text = page.extract_text()
                # Basic conversion to markdown
                # Convert headings (assuming they're in all caps)
                lines = text.split('\n')
                for line in lines:
                    if line.isupper() and len(line) < 100:  # Simple heuristic for headings
                        markdown_content.append(f"# {line}")
                    else:
                        markdown_content.append(line)
                
                markdown_content.append("\n\n")  # Add spacing between pages
            
            return "\n".join(markdown_content)
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path} to markdown: {e}")
            raise

    def process_documents(self) -> List[Dict]:
        """Process all PDFs in the Docs directory and return chunks."""
        all_chunks = []
        
        for pdf_file in DOCS_DIR.glob("*.pdf"):
            try:
                logger.info(f"Processing {pdf_file}")
                markdown_content = self.convert_pdf_to_markdown(pdf_file)
                chunks = self.markdown_splitter.split_text(markdown_content)
                
                # Add source information to each chunk
                for chunk in chunks:
                    chunk.metadata["source"] = pdf_file.name
                
                all_chunks.extend(chunks)
                logger.info(f"Successfully processed {pdf_file}")
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        return all_chunks

    def find_table_reference(self, chunk: Dict, all_chunks: List[Dict]) -> List[Dict]:
        """Find chunks containing tables referenced in the given chunk."""
        table_references = []
        content = chunk.page_content.lower()
        
        # Simple heuristic to find table references
        if "table" in content:
            for other_chunk in all_chunks:
                if other_chunk != chunk and "table" in other_chunk.page_content.lower():
                    table_references.append(other_chunk)
        
        return table_references 