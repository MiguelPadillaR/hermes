import structlog

from openai import OpenAI

from config.config import LLM_API_KEY, LLM_BASE_URL

logger = structlog.get_logger(__file__)


# --- LLM UTILS ---
def init_client(base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY):
    return OpenAI(base_url=base_url, api_key=api_key)


def load_prompt(file_path: str) -> str:
    """Reads external prompt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
