"""Vector store for reference documents."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from docx import Document

from src.config.config import get_settings

logger = logging.getLogger(__name__)


def _load_pdf_document(file_path: Path) -> str:
    """Load text content from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return ""


def _load_docx_document(file_path: Path) -> str:
    """Load text content from a DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error loading DOCX {file_path}: {e}")
        return ""


def _load_documents(docs_dir: Path) -> List[Tuple[str, str]]:
    """Load all PDF and DOCX documents from the directory.

    Returns:
        List of tuples (filename, content)
    """
    documents = []

    # Load PDF files
    for pdf_file in docs_dir.glob("*.pdf"):
        content = _load_pdf_document(pdf_file)
        if content.strip():
            documents.append((pdf_file.name, content))

    # Load DOCX files
    for docx_file in docs_dir.glob("*.docx"):
        content = _load_docx_document(docx_file)
        if content.strip():
            documents.append((docx_file.name, content))

    logger.info(f"Loaded {len(documents)} documents from {docs_dir}")
    return documents


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks with overlap."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)


def build_reference_vector_store(
    docs_dir: Optional[Path] = None,
    persist_dir: Optional[Path] = None,
    collection_name: Optional[str] = None,
    force_rebuild: bool = False,
) -> str:
    """
    Build or refresh the Chroma vector store for procurement reference docs.

    Args:
        docs_dir: Directory containing PDF/DOCX reference documents
        persist_dir: Directory where the Chroma store should be persisted
        collection_name: Chroma collection name
        force_rebuild: Whether to rebuild even if store exists

    Returns:
        Path to the vector store location
    """
    settings = get_settings()

    # Use provided values or defaults from settings
    docs_dir = docs_dir or settings.reference_docs_dir
    persist_dir = persist_dir or settings.vector_store_dir
    collection_name = collection_name or settings.vector_collection_name

    # Create persist directory if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Check if vector store already exists
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))

    if not force_rebuild and collection_name in [col.name for col in chroma_client.list_collections()]:
        logger.info(f"Vector store '{collection_name}' already exists. Use --force to rebuild.")
        return str(persist_dir)

    # Load documents
    documents = _load_documents(docs_dir)
    if not documents:
        logger.warning(f"No documents found in {docs_dir}")
        return str(persist_dir)

    # Initialize embedding model
    embedding_model = SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)

    # Create or recreate collection
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass  # Collection might not exist

    collection = chroma_client.create_collection(name=collection_name)

    # Process documents and add to vector store
    chunk_id = 0
    for doc_name, doc_content in documents:
        logger.info(f"Processing document: {doc_name}")

        # Chunk the document
        chunks = _chunk_text(doc_content, settings.reference_chunk_size, settings.reference_chunk_overlap)

        # Create embeddings for chunks
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)

        # Prepare metadata and IDs
        metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        chunk_id += len(chunks)

    logger.info(f"Successfully built vector store with {chunk_id} chunks from {len(documents)} documents")
    return str(persist_dir)


def retrieve_reference_chunks(question: str, n_results: int = 5) -> List[str]:
    """
    Retrieve relevant reference document chunks for a question.

    Args:
        question: The user's question
        n_results: Number of results to return

    Returns:
        List of relevant text chunks from reference documents
    """
    settings = get_settings()

    try:
        chroma_client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
        collection = chroma_client.get_collection(name=settings.vector_collection_name)

        # Initialize embedding model
        embedding_model = SentenceTransformer(settings.embedding_model_name, device=settings.embedding_device)

        # Encode the question
        question_embedding = embedding_model.encode([question], show_progress_bar=False)[0]

        # Search for similar chunks
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=n_results
        )

        # Extract the document chunks
        chunks = results['documents'][0] if results['documents'] else []

        logger.info(f"Retrieved {len(chunks)} reference chunks for question")
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving reference chunks: {e}")
        return []
