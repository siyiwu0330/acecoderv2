# AceCoderV2 - Adversarial Code Generation System
# Complete self-contained Docker image with all dependencies pre-installed
# Version: 2.1.0 - Added skip_step4 functionality and parquet conversion

FROM python:3.11-slim

# Set environment variables for Python and application
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install comprehensive system dependencies
RUN apt-get update && apt-get install -y \
    # Basic system tools
    git \
    curl \
    wget \
    unzip \
    vim \
    nano \
    # Build tools
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # Development tools
    pkg-config \
    # Network tools
    netcat-traditional \
    telnet \
    # Process management
    htop \
    procps \
    # Additional utilities
    tree \
    jq \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python dependencies as root for system-wide availability
RUN pip install \
    # From uv.lock equivalent packages
    # Core ML/AI packages
    numpy \
    pandas \
    matplotlib \
    seaborn \
    plotly \
    scikit-learn \
    # Web framework packages
    gradio \
    streamlit \
    flask \
    fastapi \
    # Data processing
    jsonlines \
    pyyaml \
    toml \
    # Development tools
    jupyter \
    ipython \
    # Testing tools
    pytest \
    pytest-asyncio \
    # Code quality
    black \
    isort \
    flake8 \
    # Utility packages
    tqdm \
    rich \
    click \
    fire \
    # OpenAI and LLM packages
    openai \
    anthropic \
    transformers \
    torch \
    # Evaluation packages
    datasets \
    evaluate \
    # Additional evaluation dependencies
    psutil \
    tenacity \
    # Evaluation framework - install directly
    evalplus

# Create non-root user with proper permissions
RUN useradd --create-home --shell /bin/bash acecoder \
    && usermod -aG sudo acecoder

# Switch to non-root user for security
USER acecoder
WORKDIR /home/acecoder/app

# Set application-specific environment variables
ENV PYTHONPATH=/home/acecoder/app

# Copy dependency files for reference
COPY --chown=acecoder:acecoder pyproject.toml uv.lock ./

# Note: Using direct installation instead of runtime script

# Copy application code
COPY --chown=acecoder:acecoder . .

# Note: Additional packages can be installed at runtime if needed
# pyext and vllm skipped due to compatibility/size issues
# git+https://github.com/TIGER-AI-Lab/AceCoder.git@dev removed to preserve local files

# Create all necessary directories with proper permissions
RUN mkdir -p \
    outputs \
    logs \
    eval \
    data \
    models \
    cache \
    ~/.cache/huggingface \
    ~/.cache/transformers

# Download evaluation tools (optional - skip if repositories are private)
RUN cd /tmp \
    && (git clone -b reasoning https://github.com/jdf-prog/LiveCodeBench.git || echo "LiveCodeBench not available") \
    && (git clone https://github.com/jdf-prog/AceReasonEvalKit.git || echo "AceReasonEvalKit not available") \
    && (cp -r LiveCodeBench /home/acecoder/app/eval/ 2>/dev/null || echo "LiveCodeBench not copied") \
    && (cp -r AceReasonEvalKit /home/acecoder/app/eval/ 2>/dev/null || echo "AceReasonEvalKit not copied") \
    && rm -rf /tmp/LiveCodeBench /tmp/AceReasonEvalKit 2>/dev/null || true

# Set up shell environment with useful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc \
    && echo 'alias la="ls -la"' >> ~/.bashrc \
    && echo 'alias grep="grep --color=auto"' >> ~/.bashrc \
    && echo 'alias tree="tree -C"' >> ~/.bashrc \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/home/acecoder/app:$PYTHONPATH"' >> ~/.bashrc \
    && echo 'cd /home/acecoder/app' >> ~/.bashrc

# Create useful scripts for common operations
RUN echo '#!/bin/bash\necho "=== AceCoderV2 Environment Status ==="\necho "Python version: $(python --version)"\necho "Pip packages: $(pip list | wc -l) installed"\necho "Current directory: $(pwd)"\necho "Available commands:"\necho "  - run-pipeline: Start the adversarial generation pipeline"\necho "  - run-gradio: Start the Gradio web interface"\necho "  - run-eval: Run evaluation suite"\necho "  - run-tests: Run test suite"\necho "================================="' > ~/status.sh \
    && chmod +x ~/status.sh

RUN mkdir -p ~/.local/bin \
    && echo '#!/bin/bash\npython main.py "$@"' > ~/.local/bin/run-pipeline \
    && echo '#!/bin/bash\npython integrated_gradio_app.py "$@"' > ~/.local/bin/run-gradio \
    && echo '#!/bin/bash\npython -m pytest code_eval/ "$@"' > ~/.local/bin/run-tests \
    && echo '#!/bin/bash\ncd eval && python -m pytest "$@"' > ~/.local/bin/run-eval \
    && chmod +x ~/.local/bin/run-*

# Expose all commonly used ports
EXPOSE 7860 8000 8888 5000 8080

# Health check for the main Gradio service
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health 2>/dev/null || curl -f http://127.0.0.1:7860/ 2>/dev/null || exit 1

# Default command with informative startup  
CMD echo "🚀 Starting AceCoderV2 Environment..." && \
    echo "📋 Environment ready! Available commands:" && \
    echo "  • run-gradio    - Start web interface (port 7860)" && \
    echo "  • run-pipeline  - Execute adversarial pipeline" && \
    echo "  • run-tests     - Run test suite" && \
    echo "  • run-eval      - Run evaluation" && \
    echo "💡 Local access: http://localhost:7860" && \
    echo "🌟 Public share link will be shown below (works anywhere!)" && \
    echo "🌐 Advanced: Remote access via http://YOUR_SERVER_IP:7860" && \
    echo "🔧 For shell access: docker exec -it <container> bash" && \
    echo "" && \
    python integrated_gradio_app.py

