#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data/LoveDA}"
mkdir -p "$ROOT"
cd "$ROOT"

wget -c https://zenodo.org/record/5706578/files/Train.zip
wget -c https://zenodo.org/record/5706578/files/Val.zip

unzip -n Train.zip
unzip -n Val.zip

echo "LoveDA baixada em: $ROOT"
echo "O notebook usara automaticamente as pastas images_png e masks_png."
