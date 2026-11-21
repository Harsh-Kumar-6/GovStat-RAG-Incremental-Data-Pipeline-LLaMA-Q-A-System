"""Configuration management using pydantic-settings."""
from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------------
    # Scraper
    # ---------------------------------------------------------------
    # mospi.gov.in is a client-rendered SPA: every path returns the same
    # 2644-byte HTML shell, so there is nothing for an HTML crawler to parse.
    # Content comes from the site's own JSON API, which is what we call.
    mospi_api_base: str = "https://www.mospi.gov.in/api"
    # Which API listings to pull. "latest-release" = press notes/releases,
    # "publications-reports" = reports and publications.
    sources_str: str = "latest-release,publications-reports"
    api_page_size: int = 50
    api_lang: str = "en"

    # Legacy HTML seed URLs. Retained only so existing .env files do not break;
    # unused, because the site cannot be crawled as HTML.
    seed_urls_str: str = (
        "https://mospi.gov.in/web/mospi/press-releases,"
        "https://mospi.gov.in/web/mospi/reports"
    )
    # Maximum number of API listing pages to request per run.
    max_pages_per_run: int = 5
    concurrency: int = 2
    rate_limit_delay: float = 1.5
    user_agent: str = "MoSPI-Research-Bot/1.0 (academic-research; contact@example.com)"
    respect_robots_txt: bool = True

    # Storage
    db_path: str = "data/mospi.db"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Parsing
    # Cap on tables STORED per PDF. Deliberately high: storing a parsed table
    # is cheap, and a low cap silently truncates a document mid-way -- at 25
    # the October CPI press note stopped before its state-wise table, so the
    # typed cpi_state table simply had no October data. The expensive step is
    # per-page table DETECTION, which max_table_pages bounds instead.
    max_tables_per_pdf: int = 300
    # pdfplumber's table detection is roughly quadratic in the number of ruling
    # lines on a page. On MoSPI's 200+ page statistical yearbooks (20-25 MB) a
    # single file burned over 9 minutes of CPU. Text extraction is cheap; table
    # extraction is what explodes, so it is capped by page count separately.
    max_table_pages: int = 40
    # Skip downloading files above this size entirely. The monthly press notes
    # that carry the CPI/IIP/GDP series are 300-600 KB; anything far larger is
    # a bound report whose tables are not worth the extraction cost.
    max_pdf_mb: float = 20.0

    # ETL
    # chunk_size/overlap are counted in WORDS, not characters.
    # 180 words is roughly 240 word-pieces, which stays under the 256-token
    # input limit of all-MiniLM-L6-v2. Larger chunks are silently truncated by
    # the embedding model, so the tail of each chunk would never be searchable.
    chunk_size: int = 180
    chunk_overlap: int = 40
    min_chunk_words: int = 20

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def sources(self) -> List[str]:
        return [s.strip() for s in self.sources_str.split(",") if s.strip()]

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