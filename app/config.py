from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    use_llm: bool = os.getenv("USE_LLM", "true").lower() == "true"
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    db_path: str = os.getenv("DB_PATH", "app/data/app.db")

settings = Settings()