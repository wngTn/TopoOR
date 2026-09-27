#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

partial=""
trap 'if [ -n "$partial" ]; then rm -f -- "$partial"; fi' EXIT

while read -r file_id file; do
    if [ -s "$file" ]; then
        echo "already present: $file"
        continue
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "Install uv (https://docs.astral.sh/uv/) to download the release files." >&2
        exit 1
    fi
    echo "downloading $file from Google Drive"
    mkdir -p "$(dirname "$file")"
    partial="$(mktemp "${file}.download.XXXXXX")"
    uv tool run --from gdown==5.2.0 gdown --no-cookies --output "$partial" "$file_id"
    mv -- "$partial" "$file"
    partial=""
done < scripts/gdrive_files.tsv
