#!/usr/bin/env bash

set -euo pipefail

profile_source="/usr/share/apparmor/extra-profiles/bwrap-userns-restrict"
profile_target="/etc/apparmor.d/bwrap-userns-restrict"

echo "Installing Ubuntu's bubblewrap AppArmor profile for Codex..."
sudo apt-get update
sudo apt-get install -y bubblewrap apparmor-profiles apparmor-utils

if [[ ! -f "$profile_source" ]]; then
    echo "Error: expected AppArmor profile was not installed at:" >&2
    echo "  $profile_source" >&2
    exit 1
fi

sudo install -m 0644 "$profile_source" "$profile_target"
sudo apparmor_parser -r "$profile_target"

echo "Testing bubblewrap user-namespace creation..."
if bwrap --ro-bind / / --dev /dev --proc /proc /usr/bin/true; then
    echo "Codex bubblewrap sandbox setup is working."
else
    echo "The profile was loaded, but bubblewrap is still blocked." >&2
    echo "Try rebooting, then run this script again." >&2
    exit 1
fi
