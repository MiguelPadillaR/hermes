import structlog

import numpy as np
import pandas as pd

from chromadb.utils import embedding_functions

from config.config import CLINICAL_REFERENCE_ANCHORS, LLM_MODEL_NAME
from utils.llm_utils import init_client

logger = structlog.get_logger()

# 1. CORE API CONFIGURATION
client = init_client()
MODEL_NAME =  LLM_MODEL_NAME

def verify_dataset_context(df: pd.DataFrame, threshold: float = 0.5) -> bool:
    """
    Evaluates whether the dataframe's column schema represents clinically relevant data.
    Uses semantic vector embedding distance against an anchor reference set.
    
    Args:
        df (pd.DataFrame): The incoming uploaded user dataframe.
        threshold (float): Minimum cosine similarity to accept the dataset context.
        
    Returns:
        bool: True if the file represents valid clinical metrics, False otherwise.
    """
    # 1. Clean and stringify incoming structural column names
    incoming_columns_str = ", ".join([str(col).strip().lower() for col in df.columns])
    logger.info(f"🛡️ Validating context for incoming schema structure: [{incoming_columns_str[:120]}...]")

    try:
        # 2. Instantiate the standard local embedding function (Runs 100% locally)
        local_embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # 3. Generate vectors locally (Returns a list of floating point arrays)
        user_vector_list = local_embedding_fn([incoming_columns_str])
        user_vector = np.array(user_vector_list[0])

        anchor_vector_list = local_embedding_fn(CLINICAL_REFERENCE_ANCHORS)
        anchor_vectors = [np.array(vec) for vec in anchor_vector_list]

        # 4. Measure the mathematical distance (Cosine Similarity)
        # Formula: (A • B) / (||A|| * ||B||)
        max_similarity = -1.0
        for anchor_vector in anchor_vectors:
            dot_product = np.dot(user_vector, anchor_vector)
            norm_user = np.linalg.norm(user_vector)
            norm_anchor = np.linalg.norm(anchor_vector)
            
            similarity = dot_product / (norm_user * norm_anchor)
            if similarity > max_similarity:
                max_similarity = similarity

        logger.info(f"📊 Local Semantic matching completed. Top Similarity Score: {max_similarity:.4f}")

        # 5. Evaluate against our defensive guardrail threshold
        if max_similarity >= threshold:
            logger.info("✅ Dataset verified as clinically relevant. Proceeding to pipeline processing.")
            return True
        else:
            logger.warning("❌ Rejection triggered: Layout vector is distinct from baseline patient data contexts.")
            return False

    except Exception as e:
        # Fallback guard clause in case of transient network errors or model unavailability
        logger.error(f"⚠️ Metadata embedding generation failed: {e}. Falling back to default block state.")
        logger.error(f"\t{e}")
        logger.error(f"Falling back to default block state.")
        return False