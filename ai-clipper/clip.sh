#!/usr/bin/env bash
set -e

VIDEO_DIR="video clippings"

if [ ! -d "$VIDEO_DIR" ]; then
    echo "No '$VIDEO_DIR' folder found."
    echo "Create it first:"
    echo "  mkdir -p \"$VIDEO_DIR\""
    exit 1
fi

# Collect all video files in the folder
mapfile -t files < <(find "$VIDEO_DIR" -maxdepth 1 -type f \
    \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" \
       -o -iname "*.avi" -o -iname "*.webm" -o -iname "*.m4v" \) \
    | sort)

if [ ${#files[@]} -eq 0 ]; then
    echo "No videos found in './$VIDEO_DIR/'"
    echo ""
    echo "Copy a video into that folder (via Windows Explorer) and try again:"
    echo "  C:\\User\\WEB DEV\\CLIPPER\\ai-clipper\\video clippings\\"
    exit 1
fi

echo ""
echo "Videos found in ./$VIDEO_DIR/:"
echo ""
for i in "${!files[@]}"; do
    n=$((i+1))
    printf "  [%d] %s\n" "$n" "$(basename "${files[$i]}")"
done
echo ""
read -p "Pick a number (or press Enter to cancel): " choice

if [ -z "$choice" ]; then
    echo "Cancelled."; exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#files[@]} ]; then
    echo "Invalid choice."; exit 1
fi

target="${files[$((choice-1))]}"
echo ""
echo "==> Running pipeline on: $target"
echo ""

exec bash run.sh "$target"
