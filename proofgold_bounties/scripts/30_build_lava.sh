#!/bin/bash
# 30_build_lava.sh - Clone and build Proofgold Lava from source
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

# Ensure opam environment is loaded
eval $(opam env) 2>/dev/null || true

# Clone repository
if [ -d "$LAVA_DIR" ]; then
    echo "Repository exists, checking for updates..."
    cd "$LAVA_DIR"

    # Save current commit
    OLD_COMMIT=$(git rev-parse HEAD)

    # Pull updates
    git fetch origin &>> "$BUILD_LOG"
    git pull origin master &>> "$BUILD_LOG" || echo -e "${YELLOW}WARNING${NC} (pull failed, using current version)"

    NEW_COMMIT=$(git rev-parse HEAD)

    if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
        echo -e "${YELLOW}SKIP${NC} (already up to date: $NEW_COMMIT)"
    else
        echo -e "${GREEN}Updated${NC} ($OLD_COMMIT → $NEW_COMMIT)"
    fi
else
    echo "Cloning Proofgold Lava repository..."
    git clone "$LAVA_REPO" "$LAVA_DIR" &>> "$BUILD_LOG"
    cd "$LAVA_DIR"
    echo -e "${GREEN}OK${NC}"
fi

# Record commit hash
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_DATE=$(git log -1 --format=%ci)

echo "Building from commit: $COMMIT_HASH"
echo "Commit date: $COMMIT_DATE"
echo

# Check for build system (Makefile or dune)
if [ -f "Makefile" ]; then
    BUILD_SYSTEM="make"
elif [ -f "dune-project" ] || [ -f "dune" ]; then
    BUILD_SYSTEM="dune"
else
    echo -e "${RED}ERROR: No recognized build system found${NC}"
    echo "Please check repository structure manually."
    exit 1
fi

echo "Build system: $BUILD_SYSTEM"

# Build based on detected system
echo "Starting build (this may take several minutes)..."
echo "Full build log: $BUILD_LOG"
echo

if [ "$BUILD_SYSTEM" = "make" ]; then
    # Clean previous build (safe to run)
    make clean &>> "$BUILD_LOG" || true

    # Build
    echo -n "Building with make... "
    if make &>> "$BUILD_LOG"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Check build log: $BUILD_LOG"
        exit 1
    fi

    # Find the produced binary
    BINARY_NAME=$(find . -maxdepth 1 -type f -executable | head -1)

elif [ "$BUILD_SYSTEM" = "dune" ]; then
    # Install dune if not present
    if ! command -v dune &>/dev/null; then
        echo -n "Installing dune build system... "
        opam install -y dune &>> "$BUILD_LOG"
        echo -e "${GREEN}OK${NC}"
    fi

    # Build with dune
    echo -n "Building with dune... "
    if dune build &>> "$BUILD_LOG"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Check build log: $BUILD_LOG"
        exit 1
    fi

    # Find the produced binary (usually in _build/)
    BINARY_NAME=$(find _build -type f -executable 2>/dev/null | grep -v ".so" | head -1)
fi

# Verify binary was produced
if [ -z "$BINARY_NAME" ]; then
    echo -e "${RED}ERROR: Could not find built binary${NC}"
    echo "Please check build output manually."
    exit 1
fi

echo "Binary produced: $BINARY_NAME"

# Create symlink in scripts directory for easy access
BINARY_LINK="$(pwd)/../../scripts/proofgoldlava"
if [ -L "$BINARY_LINK" ]; then
    rm "$BINARY_LINK"
fi
ln -s "$(readlink -f $BINARY_NAME)" "$BINARY_LINK"
echo "Symlink created: $BINARY_LINK"

# Create version stamp
cat > "$(pwd)/../../data/lava_build_version.txt" <<EOF
# Proofgold Lava Build Information
# Built on $(date)

Repository: $LAVA_REPO
Commit: $COMMIT_HASH
Date: $COMMIT_DATE
Build system: $BUILD_SYSTEM
Binary: $BINARY_NAME

OS: $(lsb_release -d | cut -f2-)
OCaml: $(ocaml -version 2>&1 | head -1)

Build log: $BUILD_LOG
EOF

# Test the binary
echo
echo "Testing binary..."
if [ -x "$BINARY_NAME" ]; then
    echo -n "  Checking if binary runs... "
    # Try to get version or help (different binaries have different flags)
    if $BINARY_NAME --version &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        $BINARY_NAME --version 2>&1 | head -3
    elif $BINARY_NAME -h &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    elif $BINARY_NAME help &>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        # Binary exists but we can't get version info - that's OK
        echo -e "${YELLOW}OK${NC} (executable, but can't verify version)"
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
echo "  Binary: $BINARY_NAME"
echo "  Symlink: $BINARY_LINK"
echo "  Version stamp: $(pwd)/../../data/lava_build_version.txt"
echo
echo "Next steps:"
echo "  ./scripts/40_config_proofgold.sh"
