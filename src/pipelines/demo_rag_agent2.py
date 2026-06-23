import os
import structlog

from openai import OpenAI

from config.config import (
    AGENT1_OUTPUT_REPORT_FILEPATH,
    AGENT2_OUTPUT_REPORT_FILEPATH,
    AGENT2_SYSTEM_PROMPT_FILEPATH,
    LLM_MODEL_NAME,
    )
from utils.llm_utils import init_client, load_prompt, seed_demo_knowledge_base

logger = structlog.get_logger()

# --- CONFIGURATION ---
client = init_client()
MODEL_NAME =  LLM_MODEL_NAME

# --- THE AGENT #2 EXECUTION LOGIC ---

def run_rag_pipeline():
    """
    Reads the generated report output from Agent 1, queries the vector database 
    using context metrics, and enriches it with concrete protocol guidelines.
    """
    logger.info(f"📖 Reading input report from: {AGENT1_OUTPUT_REPORT_FILEPATH}")
    if not os.path.exists(AGENT1_OUTPUT_REPORT_FILEPATH):
        fe = FileNotFoundError(f"No pre-report file found at {AGENT1_OUTPUT_REPORT_FILEPATH}. Run Agent 1 first.")
        logger.error(fe)
        raise fe
        
    with open(AGENT1_OUTPUT_REPORT_FILEPATH, "r", encoding="utf-8") as f:
        pre_report_content = f.read()
    
    # Setup our local mock vector base collection
    collection = seed_demo_knowledge_base()
    
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

    final_enriched_report = response.choices[0].message.content
    
    # Save the polished output file
    with open(AGENT2_OUTPUT_REPORT_FILEPATH, "w", encoding="utf-8") as f:
        f.write(final_enriched_report)
        
    logger.info(f"\n🎉 SUCCESS: Final Enriched Medical Report generated at '{AGENT2_OUTPUT_REPORT_FILEPATH}'")
    return final_enriched_report


if __name__ == "__main__":

    # Execute Agent 2's isolated pipeline run
    # This expects the file 'pre_report.md' generated deterministically by your Python script wrapper
    try:
        final_report = run_rag_pipeline()
        logger.info("\n======================= OUTPUT PREVIEW =======================")
        logger.info(final_report[:500] + "\n\n[... Remaining Report Content Saved To Disk ...]")
    except FileNotFoundError as e:
        logger.info(f"⚠️ Error: {e}")
