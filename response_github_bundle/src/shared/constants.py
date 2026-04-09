from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_FILE = PROJECT_ROOT / "configs" / "version.json"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".hwp", ".txt"}
