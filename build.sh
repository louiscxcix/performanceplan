#!/bin/bash
# Docker Buildx Build Script for performanceplan
# Individual project build with multi-platform support

set -e

performanceplan="performanceplan"
BUILDER_NAME="tufly-builder"
PLATFORM="linux/amd64"
PUSH=false
REGISTRY="tufly"
NO_CACHE=false

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Docker image for $performanceplan using buildx.

OPTIONS:
    --platform=PLATFORMS    Target platforms (default: $PLATFORM)
    --push                 Push image to registry after build
    --no-cache            Build without using cache
    --registry=NAME        Registry name (default: $REGISTRY)
    -h, --help            Show this help message

EXAMPLES:
    # Build for current platform
    $0

    # Build for multiple platforms and push
    $0 --platform=linux/amd64,linux/arm64 --push

    # Build without cache
    $0 --no-cache

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform=*)
            PLATFORM="${1#*=}"
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --registry=*)
            REGISTRY="${1#*=}"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check Docker and buildx
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker buildx ls | grep -q "$BUILDER_NAME"; then
    echo -e "${RED}❌ Error: Builder '$BUILDER_NAME' not found${NC}"
    echo -e "${BLUE}Run ../setup-builder.sh first${NC}"
    exit 1
fi

# Use builder
docker buildx use "$BUILDER_NAME"

# Build configuration
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo -e "${BLUE}📦 Building $performanceplan...${NC}"
echo -e "  Platform: $PLATFORM"
echo -e "  Registry: $REGISTRY"
echo -e "  Push: $PUSH"
echo ""

# Build command
BUILD_CMD="docker buildx build"
BUILD_CMD="$BUILD_CMD --builder=$BUILDER_NAME"
BUILD_CMD="$BUILD_CMD --platform=$PLATFORM"
BUILD_CMD="$BUILD_CMD --tag=$REGISTRY/$performanceplan:latest"
BUILD_CMD="$BUILD_CMD --tag=$REGISTRY/$performanceplan:$TIMESTAMP"

if [ "$NO_CACHE" = true ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
fi

if [ "$PUSH" = true ]; then
    BUILD_CMD="$BUILD_CMD --push"
else
    # For single platform, load the image
    if [[ ! "$PLATFORM" =~ "," ]]; then
        BUILD_CMD="$BUILD_CMD --load"
    fi
fi

BUILD_CMD="$BUILD_CMD ."

# Execute build
if eval "$BUILD_CMD"; then
    echo ""
    echo -e "${GREEN}✅ Successfully built $performanceplan${NC}"
    echo -e "  Image: $REGISTRY/$performanceplan:latest"
    echo -e "  Image: $REGISTRY/$performanceplan:$TIMESTAMP"
else
    echo ""
    echo -e "${RED}❌ Failed to build $performanceplan${NC}"
    exit 1
fi
