#!/bin/bash

# AceCoderV2 Docker Management Script
# Usage: ./docker-run.sh [build|run|push|clean]

set -e

IMAGE_NAME="acecoderv2"
TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"
DOCKER_HUB_USER="siyiwu0330"  # Docker Hub用户名
DOCKER_HUB_IMAGE="${DOCKER_HUB_USER}/${IMAGE_NAME}:${TAG}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build Docker image
build_image() {
    print_status "Building AceCoderV2 Docker image..."
    
    # Check if environment.yml exists
    if [[ ! -f "environment.yml" ]]; then
        print_error "environment.yml not found! Please ensure you're in the project root."
        exit 1
    fi
    
    # Build the image
    docker build -f Dockerfile.acecoder2 -t ${FULL_IMAGE_NAME} .
    
    if [[ $? -eq 0 ]]; then
        print_success "Docker image built successfully: ${FULL_IMAGE_NAME}"
        
        # Show image size
        IMAGE_SIZE=$(docker images ${FULL_IMAGE_NAME} --format "table {{.Size}}" | tail -n +2)
        print_status "Image size: ${IMAGE_SIZE}"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run Docker container
run_container() {
    print_status "Running AceCoderV2 container..."
    
    # Check if image exists
    if [[ "$(docker images -q ${FULL_IMAGE_NAME} 2> /dev/null)" == "" ]]; then
        print_warning "Image ${FULL_IMAGE_NAME} not found. Building it first..."
        build_image
    fi
    
    # Stop and remove existing container if running
    if [[ "$(docker ps -q -f name=acecoderv2-app)" ]]; then
        print_status "Stopping existing container..."
        docker stop acecoderv2-app
    fi
    
    if [[ "$(docker ps -aq -f name=acecoderv2-app)" ]]; then
        print_status "Removing existing container..."
        docker rm acecoderv2-app
    fi
    
    # Run the container
    docker run -d \
        --name acecoderv2-app \
        -p 7860:7860 \
        -p 7861:7861 \
        -e GRADIO_SERVER_PORT=7860 \
        -v acecoderv2-outputs:/home/acecoder/app/outputs \
        -v acecoderv2-logs:/home/acecoder/app/logs \
        --restart unless-stopped \
        ${FULL_IMAGE_NAME}
    
    if [[ $? -eq 0 ]]; then
        print_success "Container started successfully!"
        print_status "Access the application at:"
        print_status "  Local: http://localhost:7860"
        print_status "  Alternative: http://localhost:7861"
        print_status ""
        print_status "Container logs: docker logs -f acecoderv2-app"
        print_status "Container shell: docker exec -it acecoderv2-app bash"
        
        # Show container status
        sleep 2
        docker ps -f name=acecoderv2-app
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to push to Docker Hub
push_image() {
    print_status "Pushing to Docker Hub..."
    
    # Check if logged in to Docker Hub
    if ! docker info 2>/dev/null | grep -q "Username:"; then
        print_warning "Please login to Docker Hub first:"
        print_status "docker login"
        exit 1
    fi
    
    # Tag for Docker Hub
    docker tag ${FULL_IMAGE_NAME} ${DOCKER_HUB_IMAGE}
    
    # Push to Docker Hub
    docker push ${DOCKER_HUB_IMAGE}
    
    if [[ $? -eq 0 ]]; then
        print_success "Successfully pushed to Docker Hub: ${DOCKER_HUB_IMAGE}"
        print_status "Others can now pull with: docker pull ${DOCKER_HUB_IMAGE}"
    else
        print_error "Failed to push to Docker Hub"
        exit 1
    fi
}

# Function to clean up
clean_up() {
    print_status "Cleaning up Docker resources..."
    
    # Stop and remove container
    if [[ "$(docker ps -q -f name=acecoderv2-app)" ]]; then
        docker stop acecoderv2-app
    fi
    
    if [[ "$(docker ps -aq -f name=acecoderv2-app)" ]]; then
        docker rm acecoderv2-app
    fi
    
    # Remove image
    if [[ "$(docker images -q ${FULL_IMAGE_NAME} 2> /dev/null)" != "" ]]; then
        docker rmi ${FULL_IMAGE_NAME}
    fi
    
    # Clean up dangling images
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "AceCoderV2 Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build    Build the Docker image"
    echo "  run      Run the container (builds if needed)"
    echo "  push     Push to Docker Hub (requires login)"
    echo "  clean    Clean up Docker resources"
    echo "  status   Show container status"
    echo "  logs     Show container logs"
    echo "  shell    Access container shell"
    echo ""
    echo "Examples:"
    echo "  $0 build        # Build the image"
    echo "  $0 run          # Run the application"
    echo "  $0 push         # Push to Docker Hub"
    echo "  $0 logs         # View logs"
}

# Function to show status
show_status() {
    print_status "Container status:"
    docker ps -f name=acecoderv2-app
    
    if [[ "$(docker ps -q -f name=acecoderv2-app)" ]]; then
        print_status ""
        print_status "Container is running!"
        print_status "Access at: http://localhost:7860"
    else
        print_warning "Container is not running"
    fi
}

# Function to show logs
show_logs() {
    if [[ "$(docker ps -q -f name=acecoderv2-app)" ]]; then
        docker logs -f acecoderv2-app
    else
        print_error "Container is not running"
        exit 1
    fi
}

# Function to access shell
access_shell() {
    if [[ "$(docker ps -q -f name=acecoderv2-app)" ]]; then
        docker exec -it acecoderv2-app bash
    else
        print_error "Container is not running"
        exit 1
    fi
}

# Main script logic
case "${1:-}" in
    "build")
        build_image
        ;;
    "run")
        run_container
        ;;
    "push")
        push_image
        ;;
    "clean")
        clean_up
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "shell")
        access_shell
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
