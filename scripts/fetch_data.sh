#!/usr/bin/env bash
# Fetch the public benchmark data into data/raw (about 300 MB).
#   CMU keystroke benchmark + lab/field extension (Killourhy & Maxion), from the
#   authors' GitHub Pages repo; Balabit Mouse Dynamics Challenge.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
if [ ! -d data/raw/maxion ]; then
    git clone --depth 1 --filter=blob:none --no-checkout https://github.com/roymaxion/roymaxion.github.io data/raw/maxion
    git -C data/raw/maxion checkout HEAD -- \
        projects/keystroke-benchmark/DSL-StrongPasswordData.csv \
        projects/lab-field/FULL_lab_data.csv \
        projects/lab-field/FULL_field_data.csv
fi
if [ ! -d data/raw/balabit ]; then
    git clone --depth 1 https://github.com/balabit/Mouse-Dynamics-Challenge data/raw/balabit
fi
echo "data ready in data/raw"
