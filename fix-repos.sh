#!/bin/bash
set -e

echo "🔍 Scanning APT repository sources..."

# Backup first
backup_dir="/etc/apt/sources.list.backups"
sudo mkdir -p "$backup_dir"
timestamp=$(date +%Y%m%d_%H%M%S)
sudo cp /etc/apt/sources.list "$backup_dir/sources.list.$timestamp"
sudo cp -r /etc/apt/sources.list.d "$backup_dir/sources.list.d.$timestamp"

echo "🗂 Backups saved to $backup_dir"

# Function to check a single repo URL
check_url() {
    local url="$1"
    if curl -s --head --connect-timeout 5 "$url" | grep -q "200 OK"; then
        echo "✅ $url is reachable"
    else
        echo "❌ $url is unreachable"
        echo "$url" >> dead_repos.txt
    fi
}

# Gather and check repo URLs
echo "" > dead_repos.txt
grep -hr '^deb' /etc/apt/sources.list /etc/apt/sources.list.d | awk '{print $2}' | sort -u | while read -r repo; do
    check_url "$repo"
done

# Show results
if [[ -s dead_repos.txt ]]; then
    echo ""
    echo "⚠️  The following repositories appear to be dead or unreachable:"
    cat dead_repos.txt
    echo ""

    read -p "💬 Do you want to comment them out in your sources? [y/N]: " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        while read -r dead_url; do
            sudo sed -i "s|^deb .*${dead_url}|# [DISABLED] deb ${dead_url}|" /etc/apt/sources.list
            sudo find /etc/apt/sources.list.d -type f -exec sed -i "s|^deb .*${dead_url}|# [DISABLED] deb ${dead_url}|" {} \;
        done < dead_repos.txt
        echo "✅ Dead entries have been commented out."
    else
        echo "ℹ️ No changes made."
    fi
else
    echo "🎉 All repo sources are reachable!"
fi

