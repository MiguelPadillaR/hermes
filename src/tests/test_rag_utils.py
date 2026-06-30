from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from utils.rag_utils import chunk_document, get_knowledge_base

# =========================================================================
# 1. TESTING DOCUMENT CHUNKING
# =========================================================================

@patch("utils.rag_utils.PyPDFLoader")
def test_chunk_document_processing(mock_loader_cls):
    """
    Simulates loading a PDF document and validates that the 
    RecursiveCharacterTextSplitter generates the expected chunk array objects.
    """
    # Create a real Pydantic-valid Document instance with an empty dict for metadata
    valid_langchain_doc = Document(
        page_content="Patient protocol guidelines text content. Post-op step review criteria.",
        metadata={"source": "mock_file.pdf"}
    )
    
    # Instruct the mock instance to return our valid document list array
    mock_loader_instance = mock_loader_cls.return_value
    mock_loader_instance.load.return_value = [valid_langchain_doc]
    
    # Run local processing chunk split step
    chunks = chunk_document(path="mock_protocol.pdf", chunk_size=20, chunk_overlap=5)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0].page_content, str)


# =========================================================================
# 2. TESTING VECTOR COLLECTION SEEDING
# =========================================================================

@patch("utils.rag_utils.chromadb.PersistentClient")
@patch("utils.rag_utils.chunk_document")
@patch("utils.rag_utils.os.listdir")
def test_get_knowledge_base_seeding(mock_listdir, mock_chunk_doc, mock_chroma_client_cls):
    """
    Tests the RAG database builder mechanism. Ensures items are ingested,
    filtered (like removing bibliography text items), and written into collections.
    """
    mock_listdir.return_value = ["protocol_alpha.pdf"]
    
    # Generate valid LangChain documents for chunk simulation outputs
    doc_1 = Document(page_content="Core medical procedure directions here.", metadata={})
    doc_2 = Document(page_content="This is a segment containing bibliografia references.", metadata={})
    
    mock_chunk_doc.return_value = [doc_1, doc_2]
    
    # Mock out the internal Chroma collection calls
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0 
    
    mock_client_instance = mock_chroma_client_cls.return_value
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    
    result_collection = get_knowledge_base(reset_database=True)
    
    assert result_collection == mock_collection
    assert mock_collection.add.called
    
    # Verify our filter rules skipped the bibliography document slice
    called_kwargs = mock_collection.add.call_args[1]
    assert len(called_kwargs["documents"]) == 1
    assert "Core medical procedure" in called_kwargs["documents"][0]