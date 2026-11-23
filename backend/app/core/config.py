import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_AI_PYTHON_PATH = (
    PROJECT_ROOT / ".venv-ai" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
)


class Settings(BaseSettings):
    APP_NAME: str = "PipeAugment AI"
    APP_VERSION: str = "0.1.0"
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATASET_ROOT: Path = DEFAULT_DATASET_ROOT
    AI_PYTHON_PATH: Path = DEFAULT_AI_PYTHON_PATH

    AUGMENTED_DIR: Path = DEFAULT_DATASET_ROOT / "augmented" / "app_generated"
    JOBS_DIR: Path = DEFAULT_DATASET_ROOT / "metadata" / "jobs"
    EXPORTS_DIR: Path = DEFAULT_DATASET_ROOT / "exports"
    FINAL_EVAL_DIR: Path = PROJECT_ROOT / "experiments" / "final_evaluation"

    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_IMAGE_PIXELS: int = 25_000_000
    CORS_ALLOW_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
