#!/bin/bash

# AceCoderV2 Docker Startup Script
# This script provides easy commands to manage the Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating template..."
        cat > .env << EOF
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom model settings
DEFAULT_MODEL=gpt-4.1-mini
DEFAULT_MAX_TOKENS=4000
DEFAULT_MAX_SAMPLES=100

# Optional: Custom ports
GRADIO_PORT=7860
ALT_PORT=8000
EOF
        print_warning "Please edit .env file and add your OpenAI API key"
        return 1
    fi
    return 0
}

# Build the Docker image
build() {
    print_status "Building AceCoderV2 Docker image..."
    docker build -t acecoderv2:latest .
    print_status "Build completed successfully!"
}

# Force rebuild the Docker image (no cache)
build_fresh() {
    print_status "Force rebuilding AceCoderV2 Docker image (no cache)..."
    docker build --no-cache -t acecoderv2:latest .
    print_status "Fresh build completed successfully!"
}

# Start the application
start() {
    check_docker
    if ! check_env; then
        print_error "Please configure .env file first"
        exit 1
    fi
    
    print_status "Starting AceCoderV2 application..."
    docker compose up -d
    
    print_status "Application started successfully!"
    print_status "Local access: http://localhost:7860"
    print_status "🌟 Waiting for public share link..."
    
    # Wait for Gradio to start and extract public URL
    local public_url=""
    local attempts=0
    local max_attempts=30
    
    while [ $attempts -lt $max_attempts ] && [ -z "$public_url" ]; do
        sleep 2
        public_url=$(docker logs acecoderv2-app 2>/dev/null | grep -o "https://[a-zA-Z0-9]*\.gradio\.live" | head -1)
        attempts=$((attempts + 1))
        
        if [ $attempts -eq 15 ]; then
            print_status "⏳ Still waiting for Gradio to start..."
        fi
    done
    
    if [ -n "$public_url" ]; then
        print_status "🌍 Public share link: $public_url"
        print_status "✅ Ready! You can access from anywhere using the public link above!"
    else
        print_status "⚠️  Public link not found yet, check logs: docker compose logs -f"
        print_status "🌐 Alternative: Direct access via http://YOUR_SERVER_IP:7860"
    fi
}

# Stop the application
stop() {
    print_status "Stopping AceCoderV2 application..."
    docker compose down
    print_status "Application stopped successfully!"
}

# View logs
logs() {
    docker compose logs -f acecoderv2
}

# Clean up Docker resources
clean() {
    print_status "Cleaning up Docker resources..."
    docker compose down -v
    docker image prune -f
    print_status "Cleanup completed!"
}

# Development mode (with code mounting)
dev() {
    check_docker
    if ! check_env; then
        print_error "Please configure .env file first"
        exit 1
    fi
    
    print_status "Starting AceCoderV2 in development mode..."
    docker compose -f docker compose.yml -f docker compose.dev.yml up -d
    print_status "Development environment started!"
    print_status "Code changes will be reflected immediately"
}

# Interactive research environment
research() {
    check_docker
    if ! check_env; then
        print_error "Please configure .env file first"
        exit 1
    fi
    
    print_status "Starting AceCoderV2 research environment..."
    docker compose --profile research up -d acecoderv2-interactive
    
    print_status "Research environment started successfully!"
    print_status "Available interfaces:"
    print_status "- Gradio Interface: http://localhost:7861"
    print_status "- Jupyter Lab: http://localhost:8888"
    print_status ""
    print_status "To access the interactive shell:"
    print_status "docker exec -it acecoderv2-research bash"
}

# Enter interactive shell
shell() {
    if docker ps | grep -q acecoderv2-research; then
        print_status "Connecting to research environment shell..."
        docker exec -it acecoderv2-research bash
    elif docker ps | grep -q acecoderv2-app; then
        print_status "Connecting to production environment shell..."
        docker exec -it acecoderv2-app bash
    else
        print_error "No running containers found. Start the environment first."
        print_status "Available commands: start, research, dev"
    fi
}

# Build specific image
build_research() {
    print_status "Building AceCoderV2 research environment..."
    docker build -f Dockerfile.interactive -t acecoderv2:research .
    print_status "Research environment build completed!"
}

# Show help
help() {
    echo -e "${BLUE}AceCoderV2 Docker Management Script${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image"
    echo "  build_fresh    Force rebuild (no cache) - use this after code changes"
    echo "  build_research Build the research environment image"
    echo "  start          Start the production application"
    echo "  research       Start the interactive research environment"
    echo "  dev            Start in development mode"
    echo "  stop           Stop the application"
    echo "  restart        Restart the application"
    echo "  logs           View application logs"
    echo "  shell          Access interactive shell"
    echo "  clean          Clean up Docker resources"
    echo "  help           Show this help message"
    echo ""
    echo "Environment Modes:"
    echo "  Production:  Auto-starts web interface (port 7860)"
    echo "  Research:    Interactive shell + Jupyter + Gradio (ports 7861, 8888)"
    echo "  Development: Live code mounting for development"
    echo ""
    echo "Examples:"
    echo "  $0 build && $0 start           # Production deployment"
    echo "  $0 research                    # Research environment"
    echo "  $0 shell                       # Access running container"
    echo "  $0 logs                        # View logs"
}

# Main command handling
case "${1:-help}" in
    build)
        build
        ;;
    build_fresh)
        build_fresh
        ;;
    build_research)
        build_research
        ;;
    start)
        start
        ;;
    research)
        research
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    logs)
        logs
        ;;
    shell)
        shell
        ;;
    dev)
        dev
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        help
        ;;
    *)
        print_error "Unknown command: $1"
        help
        exit 1
        ;;
esac

