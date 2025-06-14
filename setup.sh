#!/bin/bash
set -e  # Exit on any error

# Auto-confirm mode
AUTO_YES=false
if [[ "$1" == "-y" ]]; then
    AUTO_YES=true
fi

# Confirmation function
confirm() {
    if $AUTO_YES; then
        return 0
    fi
    read -p "$1 (y/N): " choice
    case "$choice" in
        y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

# Check if running on Debian or Ubuntu
is_ubuntu=false
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        is_ubuntu=true
    fi
fi

# System updates (optional, can skip in Docker to reduce image size)
if confirm "Do you want to update system packages?"; then
    echo "Updating system packages..."
    sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y && sudo apt update && sudo apt upgrade -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt autoclean -y
fi

# Install software-properties-common (only needed for Ubuntu PPA)
if $is_ubuntu; then
    echo "Installing cmake software-properties-common..."
    sudo apt install -y cmake software-properties-common

    # Add deadsnakes PPA if it's not already present
    if ! grep -q "^deb .*\bdeadsnakes\b" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
        echo "Adding deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
    else
        echo "deadsnakes PPA already exists. Skipping..."
    fi
    echo "Updating package list again..."
    sudo apt update
fi

# Check if Python 3.10 is already installed
if command -v python3.10 >/dev/null 2>&1; then
    echo "Python 3.10 is already installed. Skipping installation."
else
    # Ask to install python3.10 (only for Ubuntu or systems without Python 3.10)
    if confirm "Do you want to install Python3.10?"; then
        echo "Installing Python 3.10 and related packages..."
        sudo apt install -y python3.10 python3.10-venv python3.10-dev cmake
    fi
fi

# Create virtual environment if it doesn't already exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
else
    echo ".venv already exists. Skipping creation."
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip (with fallback if it's not there)
echo "Upgrading pip..."
if ! command -v pip >/dev/null 2>&1; then
    python -m ensurepip
fi
pip install --upgrade pip

echo "✅ Setup complete. Python 3.10 and virtual environment ready!"
echo ""

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing app requirements..."
    pip install -r requirements.txt
    echo "✅ All requirements have been installed."
else
    echo "⚠️  requirements.txt not found. Skipping pip install."
fi

echo ""

# # Ask to launch the App right now
# if confirm "Do you want to launch the ARI5118 - Online Face Recognition App?"; then
#     # Launch backend and frontend
#     ./start.sh --all
# fi