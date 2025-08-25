#!/bin/bash

# AceCoderV2 Container Entrypoint Script
# This script runs when the container starts

set -e

echo "🚀 Starting AceCoderV2 container..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Install evalplus at runtime to avoid Docker build issues
install_evalplus() {
    print_status "Installing evalplus at runtime..."
    
    if pip show evalplus > /dev/null 2>&1; then
        print_status "✅ evalplus already installed"
        return 0
    fi
    
    # Try installing evalplus with suppressed output
    print_status "📦 Downloading and installing evalplus (this may take a moment)..."
    if pip install --user evalplus --quiet --no-warn-script-location > /dev/null 2>&1; then
        # Verify installation
        if python -c "import evalplus; print('evalplus version:', evalplus.__version__)" 2>/dev/null | grep -q "evalplus version:"; then
            local version=$(python -c "import evalplus; print(evalplus.__version__)" 2>/dev/null)
            print_status "✅ evalplus v${version} installed successfully"
            return 0
        else
            print_warning "evalplus installed but verification failed"
        fi
    else
        print_warning "Standard installation failed, trying alternative method..."
    fi
    
    # Fallback: try with --break-system-packages for newer pip versions
    if pip install --break-system-packages evalplus --quiet --no-warn-script-location > /dev/null 2>&1; then
        if python -c "import evalplus; print('evalplus version:', evalplus.__version__)" 2>/dev/null | grep -q "evalplus version:"; then
            local version=$(python -c "import evalplus; print(evalplus.__version__)" 2>/dev/null)
            print_status "✅ evalplus v${version} installed with alternative method"
            return 0
        fi
    fi
    
    print_error "❌ Failed to install evalplus. Some evaluation features may not work."
    return 1
}

# Ensure output directories exist
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p /home/acecoder/app/outputs
    mkdir -p /home/acecoder/app/logs
    mkdir -p /home/acecoder/app/cache
    mkdir -p /home/acecoder/app/data
    print_status "✅ Directories ready"
}

# Check environment variables
check_environment() {
    print_status "Checking environment..."
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        print_warning "⚠️  OpenAI API key not set or using default value"
        print_warning "Set OPENAI_API_KEY environment variable for full functionality"
    else
        print_status "✅ OpenAI API key configured"
    fi
}

# Main initialization
main() {
    print_status "=== AceCoderV2 Initialization ==="
    
    # Run setup steps
    setup_directories
    check_environment
    install_evalplus
    
    print_status "=== Initialization Complete ==="
    print_status "Starting application services..."
    
    # Check if we should run the web interface or a specific command
    if [ $# -eq 0 ]; then
        # Default: start both the main pipeline interface and gradio web UI
        print_status "🌐 Starting Gradio web interface..."
        exec python /home/acecoder/app/integrated_gradio_app.py
    else
        # Run the specified command
        print_status "🔧 Running command: $*"
        exec "$@"
    fi
}

# Run main function with all arguments
main "$@"
