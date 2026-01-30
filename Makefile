.PHONY: help build up down shell ingest query status clean logs

help: ## Show this help message
	@echo "RAG Platform - Docker Commands"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	docker-compose build

up: ## Start the container in background
	docker-compose up -d

down: ## Stop and remove containers
	docker-compose down

shell: ## Open interactive shell in container
	docker-compose run --rm rag-platform bash

ingest: ## Ingest all documents from soc/ folder
	docker-compose run --rm rag-platform rag ingest soc/

query: ## Run an interactive query (usage: make query Q="your question")
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make query Q=\"your question\""; \
		echo "Example: make query Q=\"What is Legionella?\""; \
	else \
		docker-compose run --rm rag-platform rag query "$(Q)"; \
	fi

status: ## Show index status
	docker-compose run --rm rag-platform rag index status

list: ## List all indexed documents
	docker-compose run --rm rag-platform rag index list

logs: ## Show container logs
	docker-compose logs -f rag-platform

clean: ## Remove containers, volumes, and clean data
	docker-compose down -v
	rm -rf data/vector_store/* data/graph_db/* data/metadata.db logs/*

test-connection: ## Test connection to Ollama
	docker-compose run --rm rag-platform curl -s http://host.docker.internal:11434/api/tags

rebuild: ## Rebuild and restart everything
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# Development commands
dev-shell: ## Open shell with source code mounted (for development)
	docker-compose run --rm -v $$(pwd):/app rag-platform bash

install: ## Install/update dependencies in container
	docker-compose run --rm rag-platform poetry install

# Quick start
quickstart: build ingest status ## Complete setup: build, ingest, and show status
	@echo ""
	@echo "✓ RAG Platform is ready!"
	@echo ""
	@echo "Try: make query Q=\"What is Legionella?\""
