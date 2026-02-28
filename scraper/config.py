"""Configuration management using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Scraper — stored as plain string, parsed as property
    seed_urls_str: str = (
        "https://mospi.gov.in/web/mospi/press-releases,"
        "https://mospi.gov.in/web/mospi/reports"
    )
    max_pages_per_run: int = 5
    concurrency: int = 2
    rate_limit_delay: float = 1.5
    user_agent: str = "MoSPI-Research-Bot/1.0 (academic-research; contact@example.com)"
    respect_robots_txt: bool = True

    # Storage
    db_path: str = "data/mospi.db"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # ETL
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # RAG
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = "data/processed/vector_store"
    top_k: int = 5
    temperature: float = 0.1

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def seed_urls(self) -> List[str]:
        return [u.strip() for u in self.seed_urls_str.split(",") if u.strip()]

    @property
    def raw_pdf_dir(self) -> Path:
        return Path(self.raw_data_dir) / "pdf"

    @property
    def db_path_resolved(self) -> Path:
        return Path(self.db_path)


settings = Settings()