#!/bin/bash

# AceCoderV2 Docker Hub Setup Script
# This script helps you configure and push to Docker Hub

set -e

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

# Configuration
LOCAL_IMAGE="acecoderv2:latest"
DEFAULT_HUB_USER="siyiwu0330"  # 默认用户名，可修改
VERSION="2.1.0"  # 新版本号

echo "🐳 AceCoderV2 Docker Hub 部署助手"
echo "=================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker 未运行或未安装。请启动Docker服务。"
    exit 1
fi

# Check if local image exists
if [[ "$(docker images -q ${LOCAL_IMAGE} 2> /dev/null)" == "" ]]; then
    print_error "本地镜像 ${LOCAL_IMAGE} 不存在。"
    print_status "请先运行: ./docker-run.sh build"
    exit 1
fi

# Get Docker Hub username
echo ""
read -p "请输入你的Docker Hub用户名 (默认: ${DEFAULT_HUB_USER}): " HUB_USER
HUB_USER=${HUB_USER:-$DEFAULT_HUB_USER}

# Get image version/tag
echo ""
read -p "请输入镜像版本标签 (默认: ${VERSION}): " IMAGE_TAG
IMAGE_TAG=${IMAGE_TAG:-$VERSION}

HUB_IMAGE="${HUB_USER}/acecoderv2:${IMAGE_TAG}"

print_status "配置信息:"
print_status "  本地镜像: ${LOCAL_IMAGE}"
print_status "  Docker Hub镜像: ${HUB_IMAGE}"
print_status "  镜像大小: $(docker images ${LOCAL_IMAGE} --format 'table {{.Size}}' | tail -n +2)"

echo ""
read -p "确认推送到Docker Hub? (y/N): " CONFIRM

if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    print_warning "取消推送操作"
    exit 0
fi

# Check Docker Hub login
print_status "检查Docker Hub登录状态..."
if ! docker info 2>/dev/null | grep -q "Username: ${HUB_USER}"; then
    print_warning "未登录或用户名不匹配，请先登录Docker Hub:"
    docker login
    
    # Verify login again
    if ! docker info 2>/dev/null | grep -q "Username:"; then
        print_error "登录失败，请检查用户名和密码"
        exit 1
    fi
fi

# Tag image for Docker Hub
print_status "标记镜像为: ${HUB_IMAGE}"
docker tag ${LOCAL_IMAGE} ${HUB_IMAGE}

# Push to Docker Hub
print_status "推送镜像到Docker Hub (可能需要几分钟)..."
if docker push ${HUB_IMAGE}; then
    print_success "成功推送到Docker Hub!"
    print_status ""
    print_status "🎉 部署完成! 其他用户现在可以使用:"
    print_status ""
    print_status "  # 拉取镜像"
    print_status "  docker pull ${HUB_IMAGE}"
    print_status ""
    print_status "  # 运行容器"
    print_status "  docker run -d -p 7860:7860 --name acecoderv2 ${HUB_IMAGE}"
    print_status ""
    print_status "  # 访问应用"
    print_status "  浏览器打开: http://localhost:7860"
    print_status ""
    print_status "🔗 Docker Hub链接: https://hub.docker.com/r/${HUB_USER}/acecoderv2"
else
    print_error "推送失败，请检查网络连接和权限"
    exit 1
fi

# Create usage example
cat > docker-usage-example.md << EOF
# AceCoderV2 Docker 使用示例

## 快速开始

\`\`\`bash
# 拉取镜像
docker pull ${HUB_IMAGE}

# 运行容器
docker run -d \\
  --name acecoderv2 \\
  -p 7860:7860 \\
  -e GRADIO_SERVER_PORT=7860 \\
  ${HUB_IMAGE}

# 查看日志
docker logs -f acecoderv2

# 访问应用
# 浏览器打开: http://localhost:7860
\`\`\`

## 高级使用

\`\`\`bash
# 数据持久化
docker run -d \\
  --name acecoderv2 \\
  -p 7860:7860 \\
  -v acecoderv2-outputs:/home/acecoder/app/outputs \\
  -v acecoderv2-logs:/home/acecoder/app/logs \\
  ${HUB_IMAGE}

# 自定义端口
docker run -d \\
  --name acecoderv2 \\
  -p 8080:8080 \\
  -e GRADIO_SERVER_PORT=8080 \\
  ${HUB_IMAGE}

# 访问容器shell
docker exec -it acecoderv2 bash
\`\`\`

## 镜像信息

- **大小**: ~20.8GB
- **Python**: 3.10.18
- **PyTorch**: 2.8.0+cu128
- **环境**: 完整的acecoder2 conda环境
- **包含**: 所有AI/ML依赖、Web界面、评估工具

## 支持

- GitHub: [项目仓库链接]
- Docker Hub: https://hub.docker.com/r/${HUB_USER}/acecoderv2
EOF

print_status ""
print_status "📝 使用示例已生成: docker-usage-example.md"
print_status "🎯 推送完成! AceCoderV2 现已在Docker Hub上可用"
