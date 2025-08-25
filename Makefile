# AceCoderV2 Development Makefile

.PHONY: help install install-dev test clean docker-build docker-run conda-env conda-clean

# Default target
help:
	@echo "AceCoderV2 Development Commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  conda-env       Create conda environment from environment-dev.yml"
	@echo "  install         Install project dependencies"
	@echo "  install-dev     Install in development mode"
	@echo ""
	@echo "Development:"
	@echo "  run             Start Gradio interface"
	@echo "  test            Run tests (if available)"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run Docker container"
	@echo "  docker-stop     Stop Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean           Clean temporary files"
	@echo "  conda-clean     Remove conda environment"

# Environment setup
conda-env:
	@echo "Creating conda environment..."
	conda env create -f environment-dev.yml
	@echo "Environment created. Activate with: conda activate acecoderv2-dev"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install git+https://github.com/TIGER-AI-Lab/AceCoder.git@dev

install-dev:
	@echo "Installing in development mode..."
	pip install -e .

# Development commands
run:
	@echo "Starting Gradio interface..."
	python app.py

test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then python -m pytest tests/; else echo "No tests directory found"; fi

lint:
	@echo "Running linting..."
	@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "ruff not installed, skipping"; fi

format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then black .; else echo "black not installed, skipping"; fi

# Docker commands
docker-build:
	@echo "Building Docker image..."
	./docker-start.sh build

docker-run:
	@echo "Starting Docker container..."
	./docker-start.sh start

docker-stop:
	@echo "Stopping Docker container..."
	./docker-start.sh stop

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

conda-clean:
	@echo "Removing conda environment..."
	conda env remove -n acecoderv2-dev -y

# Environment validation
check-env:
	@echo "Checking environment..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not available"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "❌ Transformers not available"
	@python -c "import openai; print(f'OpenAI: {openai.__version__}')" 2>/dev/null || echo "❌ OpenAI not available"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "❌ CUDA check failed"

# Quick setup for new developers
setup-dev: conda-env
	@echo "Activating environment and installing..."
	@echo "Run the following commands:"
	@echo "  conda activate acecoderv2-dev"
	@echo "  make install install-dev"
	@echo "  cp env.example .env"
	@echo "  # Edit .env with your API keys"
	@echo "  make run"
