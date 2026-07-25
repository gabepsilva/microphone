#!/usr/bin/env bash

set -u

missing_pactl=0
missing_ffmpeg=0

if ! command -v pactl >/dev/null 2>&1; then
    missing_pactl=1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    missing_ffmpeg=1
fi

if ((missing_pactl == 0 && missing_ffmpeg == 0)); then
    echo "pactl is already installed: $(command -v pactl)"
    echo "ffmpeg is already installed: $(command -v ffmpeg)"
    exit 0
fi

echo "The virtual audio device script requires:"
((missing_pactl)) && echo "  - pactl (audio routing)"
((missing_ffmpeg)) && echo "  - ffmpeg (WAV recording)"
echo

package_manager=""
case "$(uname -s)" in
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            package_manager="apt"
        elif command -v dnf >/dev/null 2>&1; then
            package_manager="dnf"
        elif command -v yum >/dev/null 2>&1; then
            package_manager="yum"
        elif command -v pacman >/dev/null 2>&1; then
            package_manager="pacman"
        elif command -v zypper >/dev/null 2>&1; then
            package_manager="zypper"
        else
            echo "Could not detect a supported package manager."
            echo "Install the package containing 'pactl' using your distribution's package manager."
            exit 1
        fi
        ;;
    *)
        echo "This script supports Linux only."
        exit 1
        ;;
esac

echo -n "Install the missing dependency now? [y/N] "
read -r answer
case "$answer" in
    y|Y|yes|YES)
        case "$package_manager" in
            apt)
                sudo apt-get update || exit 1
                sudo apt-get install -y pulseaudio-utils ffmpeg || exit 1
                ;;
            dnf)
                sudo dnf install -y pulseaudio-utils ffmpeg || exit 1
                ;;
            yum)
                sudo yum install -y pulseaudio-utils ffmpeg || exit 1
                ;;
            pacman)
                sudo pacman -S --needed libpulse ffmpeg || exit 1
                ;;
            zypper)
                sudo zypper install pulseaudio-utils ffmpeg || exit 1
                ;;
        esac
        ;;
    *)
        echo "Installation cancelled."
        exit 1
        ;;
esac

if command -v pactl >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
    echo "Dependencies installed successfully."
    echo "You can now run: ./virtual-audio-device.sh"
else
    echo "Installation finished, but one or more dependencies are still unavailable."
    exit 1
fi
