from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application configuration, loaded from environment variables / a .env file.

    API keys and the optional Hugging Face token are loaded from the repo-root
    .env, which is intentionally not committed to source control.
    """

    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str
    groq_api_key: SecretStr
    hf_token: SecretStr | None = None

    products_vectorstore_path: str = "data/products_vectorstore"
    dnn_weights_path: str = "data/deep_neural_network.pth"
    database_url: str = "sqlite+aiosqlite:///./data/pricer.db"
    frontend_origin: str = "http://localhost:5173"


settings = Settings()
