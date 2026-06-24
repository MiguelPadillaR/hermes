import chromadb
import os
import structlog

from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

from config.config import DOCUMENTS_DIR, HERMES_DB_PATH, LLM_API_KEY, LLM_BASE_URL

logger = structlog.get_logger(__file__)

# --- LLM UTILS ---
def init_client(base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )


def load_prompt(file_path: str) -> str:
    """Reads external prompt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
# --- RAG UTILS ---
def chunk_document(path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Read raw protocol text and yield smaller text segments.
    Args:
        path (str): The document's filepath.
        chunk_size (intç): The maximum size for each chunk.
        chunk_overlap (int): The overlap prevents sentences right on the boundaries from being cut in half.
    Returns:
        chunked_text_list?
    """
    # Load PDF documents
    logger.info(f"Uploading document: {os.path.basename(path)}")
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    chunked_text_list = text_splitter.split_documents(documents)

    logger.info(f"✅ Successfuly generated {len(chunked_text_list)} chunks from document!")
    return chunked_text_list

def seed_demo_knowledge_base():
    """
    Simulates ingesting standard medical protocols or NLM open access records 
    into our vector database store.
    """
    logger.info("📦 Seeding vector database knowledge base...")
    # --- BEST PRACTICE #1: Standardized Local Embedding & Vector Storage ---
    # We initialize a persistent ChromaDB instance simulating our external medical database
    chroma_client = chromadb.PersistentClient(path=HERMES_DB_PATH)

    # Use standard OpenAI embeddings or alternative local open-source models
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="clinical_protocols", 
        embedding_function=embedding_fn
    )
    existing_count = collection.count()
    if existing_count > 0:
        logger.info(f"💾 Found existing collection with {existing_count} chunks. Skipping re-ingestion.")
        return collection
    # Generate data for vector collection
    raw_texts = []
    metadata = []
    ids = []

    documents = os.listdir(DOCUMENTS_DIR)
    i =0
    for doc in documents:
        chunked_text_list = chunk_document(DOCUMENTS_DIR / doc)
        logger.info(f"📁 Generating vector collection data for document {doc}...")
        
        for chunked_doc in chunked_text_list:
            # Remove bibliography from Spanish docs
            if "bibliografia" in chunked_doc.page_content.lower():
                continue
            # Get text as str
            raw_texts.append(chunked_doc.page_content)
            # Retrieve and/or generate metadata
            meta = {"source": "hospital_protocols", "category": "post_op"}
            if hasattr(doc, 'metadata') and doc.metadata:
                meta.update(doc.metadata)  # captures file name/page numbers safely
            metadata.append(meta)
            # Manually generate chunk id
            ids.append(f"protocol_chunk_{i}")
            i += 1
    collection.add(
        documents=raw_texts,
        metadatas=metadata,
        ids=ids
    )
    logger.info("✅ Vector database populated successfully.")
    return collection

if __name__ == "__main__":
    documents = os.listdir(DOCUMENTS_DIR)
    for doc in documents:
        texts = chunk_document(DOCUMENTS_DIR / doc)
        print(texts[0])
