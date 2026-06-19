def load_prompt(file_path: str) -> str:
    """Reads external prompt files."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
