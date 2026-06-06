#!/bin/bash
# Download and unpack RDD2022 (13.3 GB) into $DATA/RDD2022. Check the licence on the figshare page.
# Run on a login node (needs internet). Country archives inside the main zip are unpacked too.
set -euo pipefail
ROOT="${DATA:-data}"
mkdir -p "$ROOT" && cd "$ROOT"
# figshare.com/ndownloader redirects to a signed S3 link valid for 10 s; the old
# ndownloader.figshare.com host answers scripts with 403. -c resumes an interrupted download
# (each retry gets a fresh link), and the zip only gets its final name once it is complete.
URL=https://figshare.com/ndownloader/files/38030910
if [[ ! -s RDD2022.zip ]]; then
    rm -f RDD2022.zip
    wget -c --tries=20 --waitretry=10 -O RDD2022.zip.part "$URL"
    mv RDD2022.zip.part RDD2022.zip
fi
unzip -q -n RDD2022.zip -d RDD2022
find RDD2022 -name "*.zip" -print0 | while IFS= read -r -d '' z; do unzip -q -n "$z" -d "$(dirname "$z")"; done
find RDD2022 -path "*/train/images/*.jpg" | awk -F/ '{print $(NF-3)}' | sort | uniq -c
