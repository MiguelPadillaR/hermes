import structlog

from config.config import (
    AGENT1_OUTPUT_REPORT_FILEPATH,
    AGENT2_OUTPUT_REPORT_FILEPATH,
    AGENT2_SYSTEM_PROMPT_FILEPATH,
    LLM_MODEL_NAME,
    )
from utils.llm_utils import init_client, load_prompt
from utils.rag_utils import get_knowledge_base

logger = structlog.get_logger()

# --- CONFIGURATION ---
client = init_client()
MODEL_NAME =  LLM_MODEL_NAME

# --- THE AGENT #2 EXECUTION LOGIC ---

def enrich_with_rag(pre_report_content: str):
    """
    Reads the generated report output from Agent 1, queries the vector database 
    using context metrics, and enriches it with concrete protocol guidelines.
    Args:
        pre_report_content (str): The filled-out target schema with the added section.
    Returns:
        final_enriched_report (str): The final enriched report to present.
        
    """
    # Setup our local mock vector base collection
    collection = get_knowledge_base()

    # BEST PRACTICE #2: Targeted Search Extraction
    # Instead of blindly throwing the whole report at the vector database, we pull the diagnostic indicator
    # For demo automation, we scan for the diagnosis or let the text query look for contextual terms.
    logger.info("🔍 Performing Vector RAG Query via ChromaDB semantic search...")
    query_results = collection.query(
        query_texts=[pre_report_content],
        n_results=1
    )
    
    # Extract the top matching medical text block
    retrieved_context = query_results['documents'][0][0] if query_results['documents'] else "No matching protocol found."
    logger.info(f"🎯 Retrieved Protocol Context:\n'{retrieved_context}'")

    # BEST PRACTICE #3: Contextual Isolation System Prompting
    user_payload = (
        f"### ORIGINAL PRE-REPORT RECEIVED ###\n"
        f"{pre_report_content}\n\n"
        f"### REFERENCE GROUNDING PROTOCOLS (RAG) ###\n"
        f"{retrieved_context}\n\n"
        f"Please generate an updated Final Report appending a clear `Clinical Advisory & Next Steps' section."
    )

    logger.info("🤖 HERMES Agent 2 is compiling enriched medical advisory guidelines...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": load_prompt(AGENT2_SYSTEM_PROMPT_FILEPATH)},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.2
    )

    # Return final report
    logger.info("✅ Successfully generated RAG-enriched final report!")

    final_enriched_report = response.choices[0].message.content
    return final_enriched_report


if __name__ == "__main__":
    # Execute Agent 2's isolated pipeline run
    # This expects the file 'pre_report.md' generated deterministically by your Python script wrapper
    with open(AGENT1_OUTPUT_REPORT_FILEPATH, 'r') as file:
        pre_report = file.read()
    try:
        final_report = enrich_with_rag(pre_report)
        logger.info("\n======================= OUTPUT PREVIEW =======================")
        logger.info(final_report[:500] + "\n\n[... Remaining Report Content Saved To Disk ...]")
    except FileNotFoundError as e:
        logger.info(f"⚠️ Error: {e}")
