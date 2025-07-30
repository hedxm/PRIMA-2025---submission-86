from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from typing import List, Optional # Added Optional and List for type hinting

@tool
def read_pdf_chunked(
    path: str, 
    chunk_size: int = 1200, 
    chunk_overlap: int = 100, 
    max_chunks: Optional[int] = None
) -> List[str]:
    """
    Reads a PDF using PyPDFLoader, concatenates all page text, 
    and splits it into chunks using RecursiveCharacterTextSplitter.

    Args:
        path: Path to the PDF file.
        chunk_size: Target size for text chunks.
        chunk_overlap: Overlap between chunks.
        max_chunks: Optional limit on the number of chunks returned.

    Returns:
        A list of text chunks extracted from the PDF. Returns empty list on error.
    """
    print(f"INFO: Starting PDF processing (read_pdf_chunked) for: {path}")
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        
        if not documents:
             print(f"⚠️ Warning: PyPDFLoader returned no documents for: {path}")
             return []

        # Combine all text content - ensure page_content exists and is string
        all_text = "\n\n".join(
            doc.page_content for doc in documents if doc.page_content and isinstance(doc.page_content, str)
        ).strip()

        if not all_text:
            print(f"⚠️ Warning: No text content extracted by PyPDFLoader from PDF {path}")
            return []

        # print(f"DEBUG: Total text length before splitting: {len(all_text)}") # Optional Debug

        # Chunk it intelligently
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Keep default separators
            length_function=len,
            is_separator_regex=False
        )
        chunks = splitter.split_text(all_text)
        
        if not chunks:
            print(f"⚠️ Warning: Text splitting resulted in no chunks for PDF {path}")
            return []
            
        final_chunks = chunks[:max_chunks] if max_chunks is not None else chunks
        print(f"INFO: Successfully chunked PDF into {len(final_chunks)} chunks using PyPDFLoader strategy.")
        return final_chunks

    except FileNotFoundError:
        print(f"🚨 ERROR: PDF file not found at: {path}")
        return []
    except Exception as e:
        print(f"🚨 ERROR: Failed during PyPDFLoader processing or splitting for PDF '{path}': {e}")
        # import traceback # Optional for detailed debug
        # traceback.print_exc()
        return []
