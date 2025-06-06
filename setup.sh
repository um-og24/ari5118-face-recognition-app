#!/bin/bash
set -e  # Exit on any error

# Initialize auto-response variable
AUTO_RESPONSE=""

# Parse command-line arguments
while getopts "yn" opt; do
    case $opt in
        y|Y) AUTO_RESPONSE="y" ;;
        n|N) AUTO_RESPONSE="n" ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Function to ask user for confirmation
confirm() {
    if [ -n "$AUTO_RESPONSE" ]; then
        # If AUTO_RESPONSE is set, return based on its value
        if [ "$AUTO_RESPONSE" = "y" ]; then
            echo "$1 (auto: Yes)"
            return 0
        else
            echo "$1 (auto: No)"
            return 1
        fi
    else
        # Interactive prompt
        read -p "$1 (y/N): " choice
        case "$choice" in
            y|Y) return 0 ;;
            *) return 1 ;;
        esac
    fi
}

# Check if running on Debian or Ubuntu
is_ubuntu=false
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        is_ubuntu=true
    fi
fi

# System updates
if confirm "Do you want to update system packages?"; then
    echo "Updating system packages..."
    sudo apt autoremove -y && sudo apt clean -y && sudo apt autoclean -y && sudo apt update && sudo apt upgrade -y && sudo apt full-upgrade -y && sudo apt autoremove -y && sudo apt autoclean -y
fi

echo "Installing software-properties-common..."
sudo apt install -y software-properties-common

# Add deadsnakes PPA (Ubuntu only)
if $is_ubuntu; then
    # Add deadsnakes PPA if it's not already present
    if ! grep -q "^deb .*\bdeadsnakes\b" /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null; then
        echo "Adding deadsnakes PPA..."
        sudo add-apt-repository -y ppa:deadsnakes/ppa
    else
        echo "deadsnakes PPA already exists. Skipping..."
    fi
fi

echo "Updating package list again..."
sudo apt update

# Ask to install python3.10
if confirm "Do you want to install Python3.10?"; then
    echo "Installing Python 3.10 and related packages..."
    sudo apt install -y python3.10 python3.10-venv python3.10-dev cmake

    # Create virtual environment if it doesn't already exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3.10 -m venv .venv
    else
        echo ".venv already exists. Skipping creation."
    fi
else
    # Create virtual environment if it doesn't already exist
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python -m venv .venv
    else
        echo ".venv already exists. Skipping creation."
    fi
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