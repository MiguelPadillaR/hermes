import chromadb
import structlog

from chromadb.utils import embedding_functions

from config.config import HERMES_DB_PATH

logger = structlog.get_logger(__file__)

# --- LLM UTILS ---
def load_prompt(file_path: str) -> str:
    """Reads external prompt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
# --- RAG UTILS ---
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

    # Example highly structured documentation chunks
    documents = [
        "Protocol HF-2026: In instances of Acute Decompensated Heart Failure, check fluid log balance. If heart rate spikes natively over 110 bpm, scale metrics to strict oxygen logs and evaluate saturation.",
        "Protocol PNEU-09: Community-Acquired Pneumonia pathways dictate maintaining constant pulse oximetry tracking. Watch for sudden skin temperature drops or spikes breaching 38.5 Celsius.",
        "Surgical Protocol WHI-102: Post-Op Whipple Procedure requires vigilant tracking of abdominal drains. Rapid bleeding, pale complexion, and rapid drops in arterial blood pressure signify critical internal hemorrhaging requiring emergency surgical re-exploration."
    ]
    
    metadatas = [
        {"source": "Cardiology_Guidelines_2026"},
        {"source": "Infectious_Disease_Manual"},
        {"source": "Gastrointestinal_Surgery_Standard"}
    ]
    
    ids = ["doc_hf_2026", "doc_pneu_09", "doc_whi_102"]
    
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("✅ Vector database populated successfully.")
    return collection
