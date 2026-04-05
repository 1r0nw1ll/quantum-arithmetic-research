#!/bin/bash
# 10_install_deps_debian_fast.sh - Fast dependency installation using system packages
# Uses system OCaml instead of opam compilation (much faster)

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

INSTALL_LOG="./data/install_deps_fast.log"

echo "=========================================="
echo "Proofgold Lava: Fast Dependency Install"
echo "=========================================="
echo
echo "Using system packages (no opam compilation)"
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

# Install DBM/GDBM libraries
echo -n "Installing DBM libraries... "
sudo apt-get install -y \
    libgdbm-dev \
    libdb-dev \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Install OCaml and related tools via system packages
echo -n "Installing OCaml ecosystem... "
sudo apt-get install -y \
    ocaml \
    ocaml-native-compilers \
    ocaml-findlib \
    ocaml-compiler-libs \
    libzarith-ocaml-dev \
    libnum-ocaml-dev \
    camlp5 \
    dune \
    &>> "$INSTALL_LOG"
echo -e "${GREEN}OK${NC}"

# Verify installations
echo
echo "Verifying installations..."
echo -n "  OCaml version: "
ocaml -version 2>&1 | head -1
echo -n "  Zarith: "
if ocamlfind query zarith &>/dev/null; then
    echo -e "${GREEN}installed$(NC) ($(ocamlfind query zarith -format '%v'))"
else
    echo -e "${YELLOW}checking system...${NC}"
    dpkg -l | grep zarith | awk '{print $2, $3}'
fi
echo -n "  Dune: "
dune --version 2>&1 || echo "not found (may not be needed)"

# Create version stamp
cat > ./data/build_versions.txt <<EOF
# Dependency versions installed on $(date)

OCaml: $(ocaml -version 2>&1 | head -1)
Zarith: $(dpkg -l | grep libzarith | awk '{print $2, $3}' | head -1)
GCC: $(gcc --version | head -1)
OS: $(lsb_release -d | cut -f2- 2>/dev/null || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)

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
