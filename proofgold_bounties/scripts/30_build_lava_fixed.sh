#!/bin/bash
# 30_build_lava_fixed.sh - Build Proofgold Lava following official INSTALL instructions
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

LAVA_REPO="https://github.com/ckaliszyk/proofgold-lava"
LAVA_DIR="$(pwd)/data/proofgold-lava"
BUILD_LOG="$(pwd)/data/lava_build.log"

echo "=========================================="
echo "Proofgold Lava: Build from Source"
echo "=========================================="
echo

# Ensure repository exists
if [ ! -d "$LAVA_DIR" ]; then
    echo "Cloning Proofgold Lava repository..."
    git clone "$LAVA_REPO" "$LAVA_DIR" &>> "$BUILD_LOG"
    echo -e "${GREEN}OK${NC}"
fi

cd "$LAVA_DIR"

# Record commit hash
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_DATE=$(git log -1 --format=%ci)

echo "Building from commit: $COMMIT_HASH"
echo "Commit date: $COMMIT_DATE"
echo

# Following official INSTALL instructions:
# 1. Build pgc library
echo "=========================================="
echo "Step 1: Building pgc library..."
echo "=========================================="

cd pgc

# Clean previous build
make clean &>> "$BUILD_LOG" || true

# Build pgc
echo -n "Building pgc... "
if make &>> "$BUILD_LOG"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Check build log: $BUILD_LOG"
    tail -50 "$BUILD_LOG"
    exit 1
fi

# 2. Build client
echo
echo "=========================================="
echo "Step 2: Building Proofgold Lava client..."
echo "=========================================="

cd ../client

# Clean previous build first (before configure)
make clean &>> "$BUILD_LOG" || true

# Run configure (this generates src/config.ml)
echo -n "Configuring client... "
if ./configure &>> "$BUILD_LOG"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Check build log: $BUILD_LOG"
    tail -50 "$BUILD_LOG"
    exit 1
fi

# Build client (don't clean again!)
echo -n "Building client... "
if make &>> "$BUILD_LOG"; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    echo "Check build log: $BUILD_LOG"
    tail -50 "$BUILD_LOG"
    exit 1
fi

# Find the binary
echo
echo "Looking for built binary..."
BINARY_NAME=""

# Common binary names in Proofgold
for name in proofgold proofgoldlava pgc client; do
    if [ -f "$name" ] && [ -x "$name" ]; then
        BINARY_NAME="$name"
        break
    fi
done

# Check in bin/ directory
if [ -z "$BINARY_NAME" ] && [ -d "bin" ]; then
    BINARY_NAME=$(find bin -maxdepth 1 -type f -executable | head -1)
fi

# Check in src/ directory
if [ -z "$BINARY_NAME" ] && [ -d "src" ]; then
    BINARY_NAME=$(find src -maxdepth 1 -type f -executable | head -1)
fi

# If still not found, list what was created
if [ -z "$BINARY_NAME" ]; then
    echo -e "${YELLOW}Could not auto-detect binary${NC}"
    echo "Files created by build:"
    find . -newer "$BUILD_LOG" -type f -executable
    echo
    echo "Please check the build output and README.md"
    exit 1
fi

echo "Binary found: $BINARY_NAME"

# Get absolute path
BINARY_PATH=$(readlink -f "$BINARY_NAME")

# Create symlink in scripts directory for easy access
BINARY_LINK="$(readlink -f $(pwd)/../../scripts/proofgoldlava)"
if [ -L "$BINARY_LINK" ]; then
    rm "$BINARY_LINK"
fi
ln -s "$BINARY_PATH" "$BINARY_LINK"
echo "Symlink created: $BINARY_LINK → $BINARY_PATH"

# Create version stamp
cat > "$(pwd)/../../data/lava_build_version.txt" <<EOF
# Proofgold Lava Build Information
# Built on $(date)

Repository: $LAVA_REPO
Commit: $COMMIT_HASH
Date: $COMMIT_DATE
Binary: $BINARY_PATH

OS: $(lsb_release -d | cut -f2- 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
OCaml: $(ocaml -version 2>&1 | head -1)

Build log: $BUILD_LOG
EOF

# Test the binary
echo
echo "Testing binary..."
if [ -x "$BINARY_PATH" ]; then
    echo -n "  Checking if binary runs... "
    # Try to get version or help
    if timeout 5 $BINARY_PATH --version &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        $BINARY_PATH --version 2>&1 | head -3
    elif timeout 5 $BINARY_PATH -h &>/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    elif timeout 5 $BINARY_PATH help &>/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        # Binary exists but we can't get version info - that's OK
        echo -e "${YELLOW}OK${NC} (executable, version check unavailable)"
    fi
else
    echo -e "${RED}ERROR: Binary is not executable${NC}"
    exit 1
fi

echo
echo "=========================================="
echo -e "${GREEN}Proofgold Lava built successfully!${NC}"
echo "=========================================="
echo
echo "Build information:"
echo "  Commit: $COMMIT_HASH"
echo "  Binary: $BINARY_PATH"
echo "  Symlink: $BINARY_LINK"
echo "  Version stamp: $(pwd)/../../data/lava_build_version.txt"
echo
echo "Next steps:"
echo "  ./scripts/40_config_proofgold.sh"
