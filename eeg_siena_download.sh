#!/usr/bin/env bash
# Download Siena Scalp EEG Database from PhysioNet.
# Run from signal_experiments/ root.
# Data lands at: archive/phase_artifacts/phase2_data/eeg/siena/

set -euo pipefail

BASE="https://physionet.org/files/siena-scalp-eeg/1.0.0"
DEST="archive/phase_artifacts/phase2_data/eeg/siena"
mkdir -p "$DEST"

PATIENTS=(PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17)

for P in "${PATIENTS[@]}"; do
    echo "=== Downloading $P ==="
    mkdir -p "$DEST/$P"
    wget -r -N -c -np -nH --cut-dirs=4 \
         --directory-prefix="$DEST/$P" \
         --accept="*.edf,*.txt" \
         --reject="index.html*" \
         "$BASE/$P/" 2>&1 | tail -3
done

echo ""
echo "Done. Summary:"
for P in "${PATIENTS[@]}"; do
    edf_n=$(ls "$DEST/$P"/*.edf 2>/dev/null | wc -l)
    txt_n=$(ls "$DEST/$P"/*.txt 2>/dev/null | wc -l)
    sz=$(du -sh "$DEST/$P" 2>/dev/null | cut -f1)
    echo "  $P: ${edf_n} edfs, ${txt_n} txt, $sz"
done
