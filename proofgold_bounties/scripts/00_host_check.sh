#!/bin/bash
# 00_host_check.sh - System verification before Proofgold Lava installation
# Safe to run multiple times (idempotent)

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Proofgold Lava: Host Check"
echo "=========================================="
echo

# Check OS
echo -n "Checking OS... "
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]] || [[ "$ID" == "debian" ]]; then
        echo -e "${GREEN}OK${NC} ($PRETTY_NAME)"
    else
        echo -e "${YELLOW}WARNING${NC} (Not Debian/Ubuntu, may have issues)"
        echo "  Detected: $PRETTY_NAME"
    fi
else
    echo -e "${RED}FAIL${NC} (Cannot detect OS)"
    exit 1
fi

# Check architecture
echo -n "Checking architecture... "
ARCH=$(uname -m)
if [[ "$ARCH" == "x86_64" ]]; then
    echo -e "${GREEN}OK${NC} (x86_64)"
elif [[ "$ARCH" == "aarch64" ]]; then
    echo -e "${YELLOW}WARNING${NC} (ARM64, may need source builds)"
else
    echo -e "${RED}FAIL${NC} ($ARCH not recommended)"
fi

# Check disk space (need at least 50GB for Litecoin + Proofgold sync)
echo -n "Checking disk space... "
AVAIL_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAIL_GB" -ge 50 ]; then
    echo -e "${GREEN}OK${NC} (${AVAIL_GB}GB available)"
else
    echo -e "${YELLOW}WARNING${NC} (Only ${AVAIL_GB}GB available, recommend 50GB+)"
fi

# Check memory
echo -n "Checking memory... "
MEM_GB=$(free -g | awk '/Mem:/ {print $2}')
if [ "$MEM_GB" -ge 4 ]; then
    echo -e "${GREEN}OK${NC} (${MEM_GB}GB RAM)"
else
    echo -e "${YELLOW}WARNING${NC} (Only ${MEM_GB}GB RAM, recommend 4GB+)"
fi

# Check internet connectivity
echo -n "Checking internet... "
if ping -c 1 8.8.8.8 &>/dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAIL${NC} (No internet connection)"
    exit 1
fi

# Check if running as root (we don't want this)
echo -n "Checking user privileges... "
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}FAIL${NC} (Do not run as root, use sudo when needed)"
    exit 1
else
    echo -e "${GREEN}OK${NC} (non-root user)"
fi

# Check sudo access (will need it for installs)
echo -n "Checking sudo access... "
if sudo -n true 2>/dev/null; then
    echo -e "${GREEN}OK${NC} (passwordless sudo)"
elif sudo -v; then
    echo -e "${GREEN}OK${NC} (sudo available)"
else
    echo -e "${RED}FAIL${NC} (sudo required for installation)"
    exit 1
fi

echo
echo "=========================================="
echo -e "${GREEN}Host check complete!${NC}"
echo "=========================================="
echo
echo "Next steps:"
echo "  ./scripts/10_install_deps_debian.sh"
