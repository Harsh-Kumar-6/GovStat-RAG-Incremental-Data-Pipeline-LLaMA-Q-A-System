.PHONY: crawl parse etl tables index models smoke up down lint test clean help

## Run the full scraper crawl
crawl:
	python -m scraper crawl

## Parse / download PDFs and extract text/tables
parse:
	python -m scraper parse

## Run the ETL pipeline (transform + chunk + validate)
etl:
	python -m pipeline.run

## Build typed SQL tables (cpi_state) from the parsed table JSON
tables:
	python -m pipeline.tables_to_sql

## List the Gemini models this API key can reach (costs 1 API request)
models:
	python -m agents models

## Prove the API key works with one minimal call (costs 1 API request)
smoke:
	python -m agents smoke

## Build the FAISS vector index
index:
	python -c "from rag.indexer import get_store; s = get_store(); s.build()"

## Start all Docker services (API + UI + Ollama)
up:
	docker compose up --build -d
	@echo "UI: http://localhost:8501"
	@echo "API: http://localhost:8000/docs"

## Run scraper service in Docker
crawl-docker:
	docker compose --profile scraper up scraper

## Run pipeline service in Docker
etl-docker:
	docker compose --profile pipeline up pipeline

## Stop all Docker services
down:
	docker compose down

## Pull LLaMA3 model into Ollama
pull-model:
	docker compose exec ollama ollama pull llama3

## Lint & format
lint:
	black scraper pipeline rag agents tests
	isort scraper pipeline rag agents tests
	mypy scraper pipeline rag agents

## Run tests
test:
	pytest tests/ -v

## Show run summary
report:
	python -m scraper report

## Clean generated data
clean:
	rm -rf data/processed data/raw/pdf datasets/catalog.json

help:
	@grep -E '^##' Makefile | sed 's/## //'