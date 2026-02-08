.PHONY: install install-dev install-cli test test-v lint clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install in development mode
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "\nActivate with: source .venv/bin/activate"

install-cli: ## Install with LLM support (requires ANTHROPIC_API_KEY)
	python -m venv .venv
	.venv/bin/pip install -e ".[dev,cli]"
	@echo "\nActivate with: source .venv/bin/activate"

test: ## Run all tests
	.venv/bin/python -m pytest tests/

test-v: ## Run all tests (verbose)
	.venv/bin/python -m pytest tests/ -v

lint: ## Run basic checks
	.venv/bin/python -m py_compile src/stigmergy/__init__.py
	@echo "Compile check passed"

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
