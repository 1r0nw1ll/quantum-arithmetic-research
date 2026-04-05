#!/bin/bash
# 10_install_deps_debian.sh - Install all dependencies for Proofgold Lava
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_LOG="./data/install_deps.log"

echo "=========================================="
echo "Proofgold Lava: Dependency Installation"
echo "=========================================="
echo
echo "Installing system packages..."
echo "(This requires sudo and may take several minutes)"
echo

# Update package lists
echo -n "Updating package lists... "
sudo apt-get update &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install build essentials
echo -n "Installing build tools... "
sudo apt-get install -y \
    build-essential \
    make \
    gcc \
    g++ \
    m4 \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install GMP (required for Zarith)
echo -n "Installing GMP libraries... "
sudo apt-get install -y \
    libgmp-dev \
    libgmpxx4ldbl \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install DBM/GDBM libraries (Lava uses Unix DBM-style storage)
echo -n "Installing DBM libraries... "
sudo apt-get install -y \
    libgdbm-dev \
    libdb-dev \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install OCaml via system package manager (opam will upgrade if needed)
echo -n "Installing OCaml base... "
sudo apt-get install -y \
    ocaml \
    ocaml-native-compilers \
    ocaml-findlib \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install opam (OCaml package manager)
echo -n "Installing opam... "
if command -v opam &>/dev/null; then
    echo -e "${YELLOW}SKIP${NC} (already installed)"
else
    sudo apt-get install -y opam &>> "$INSTALL_LOG"
    echo -e "${GREEN}OK${NC}"
fi

# Initialize opam (safe to run multiple times)
echo -n "Initializing opam... "
if [ -d "$HOME/.opam" ]; then
    echo -e "${YELLOW}SKIP${NC} (already initialized)"
else
    opam init -y --disable-sandboxing &>> "$INSTALL_LOG"
    eval $(opam env)
    echo -e "${GREEN}OK${NC}"
fi

# Ensure opam environment is loaded
eval $(opam env) 2>/dev/null || true

# Install OCaml compiler via opam (latest stable)
echo -n "Installing OCaml compiler via opam... "
OCAML_VERSION="4.14.1"  # Stable version, adjust if needed
if opam switch list | grep -q "$OCAML_VERSION"; then
    echo -e "${YELLOW}SKIP${NC} (already installed)"
else
    opam switch create "$OCAML_VERSION" &>> "$INSTALL_LOG"
    eval $(opam env)
    echo -e "${GREEN}OK${NC}"
fi

# Install Zarith (arbitrary precision integers, required by Proofgold)
echo -n "Installing Zarith library... "
if opam list | grep -q "zarith"; then
    echo -e "${YELLOW}SKIP${NC} (already installed)"
else
    opam install -y zarith &>> "$INSTALL_LOG"
    echo -e "${GREEN}OK${NC}"
fi

# Install other useful OCaml libraries
echo -n "Installing additional OCaml libraries... "
opam install -y \
    num \
    cryptokit \
    &>> "$INSTALL_LOG" || echo -e "${YELLOW}WARNING${NC} (some optional libs failed)"
echo -e "${GREEN}OK${NC}"

# Verify installations
echo
echo "Verifying installations..."
echo -n "  OCaml version: "
ocaml -version 2>&1 | head -1
echo -n "  opam version: "
opam --version
echo -n "  Zarith: "
if opam list | grep -q "zarith"; then
    echo -e "${GREEN}installed${NC}"
else
    echo -e "${RED}MISSING${NC}"
    exit 1
fi

# Create version stamp
cat > ./data/build_versions.txt <<EOF
# Dependency versions installed on $(date)

OCaml: $(ocaml -version 2>&1 | head -1)
opam: $(opam --version)
Zarith: $(opam list | grep zarith | awk '{print $2}')
GCC: $(gcc --version | head -1)
OS: $(lsb_release -d | cut -f2-)

Installation log: $INSTALL_LOG
EOF

echo
echo "=========================================="
echo -e "${GREEN}Dependencies installed successfully!${NC}"
echo "=========================================="
echo
echo "Version stamp saved to: ./data/build_versions.txt"
echo "Full installation log: $INSTALL_LOG"
echo
echo "Next steps:"
echo "  ./scripts/20_setup_litecoin.sh"
