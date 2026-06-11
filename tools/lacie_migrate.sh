#!/usr/bin/env bash
# lacie_migrate.sh — Move large data to LaCie, leave symlinks in place
# Run as: bash tools/lacie_migrate.sh
set -euo pipefail

LACIE="/Volumes/lacie/signal_experiments_offload"
REPO="/Users/player3/signal_experiments"

echo "=== LaCie Migration Script ==="
echo "Destination: $LACIE"
echo "Source repo: $REPO"
echo ""

# Abort if LaCie not writable
touch "$LACIE/.write_test" 2>/dev/null || { echo "ERROR: LaCie not writable. Check Full Disk Access for Terminal in System Preferences → Privacy & Security."; exit 1; }
rm -f "$LACIE/.write_test"
echo "✓ LaCie is writable"

# ─── Helper: move dir and leave symlink ──────────────────────────────────────
move_and_link() {
    local src="$1"
    local dst="$2"
    if [ -L "$src" ]; then
        echo "  SKIP (already symlink): $src"
        return
    fi
    if [ ! -e "$src" ]; then
        echo "  SKIP (not found): $src"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    echo "  Moving: $src → $dst"
    rsync -a --remove-source-files "$src/" "$dst/"
    find "$src" -type d -empty -delete 2>/dev/null || true
    rm -rf "$src" 2>/dev/null || true
    ln -s "$dst" "$src"
    echo "  ✓ Symlink: $src → $dst"
}

# ─── Helper: move single file and leave symlink ───────────────────────────────
move_file_and_link() {
    local src="$1"
    local dst_dir="$2"
    local dst="$dst_dir/$(basename "$src")"
    if [ -L "$src" ]; then
        echo "  SKIP (already symlink): $src"
        return
    fi
    if [ ! -e "$src" ]; then
        echo "  SKIP (not found): $src"
        return
    fi
    mkdir -p "$dst_dir"
    echo "  Moving file: $src → $dst"
    rsync -a "$src" "$dst_dir/"
    rm -f "$src"
    ln -s "$dst" "$src"
    echo "  ✓ Symlink: $src → $dst"
}

# ─── Helper: move dir without symlink (Downloads — no code references these) ──
move_only() {
    local src="$1"
    local dst="$2"
    if [ ! -e "$src" ]; then
        echo "  SKIP (not found): $src"
        return
    fi
    mkdir -p "$(dirname "$dst")"
    echo "  Moving (no symlink): $src → $dst"
    if [ -d "$src" ]; then
        rsync -a --remove-source-files "$src/" "$dst/"
        find "$src" -type d -empty -delete 2>/dev/null || true
        rm -rf "$src" 2>/dev/null || true
    else
        rsync -a "$src" "$dst"
        rm -f "$src"
    fi
    echo "  ✓ Moved"
}

echo ""
echo "=== TIER 1: Corpus data ==="
move_and_link "$REPO/corpus/pepe_pose"          "$LACIE/corpus/pepe_pose"
move_and_link "$REPO/corpus/cmu_mocap_zhou2019" "$LACIE/corpus/cmu_mocap_zhou2019"
move_and_link "$REPO/corpus/modelnet40"         "$LACIE/corpus/modelnet40"
move_and_link "$REPO/corpus/cmu_mocap_asfamc"   "$LACIE/corpus/cmu_mocap_asfamc"

echo ""
echo "=== TIER 1: Experiment caches ==="
move_and_link "$REPO/experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13" \
              "$LACIE/experiments/qa_ml/cache_pepe_ch4_pose1_table_4_2_finetuned_full13"
move_and_link "$REPO/experiments/qa_ml/pepe_ch2_rot3_rebuilt" \
              "$LACIE/experiments/qa_ml/pepe_ch2_rot3_rebuilt"
move_and_link "$REPO/experiments/qa_ml/cache_full_psp" \
              "$LACIE/experiments/qa_ml/cache_full_psp"

echo ""
echo "=== TIER 1: QA lab data ==="
move_and_link "$REPO/qa_lab/data/rruff_raman"         "$LACIE/qa_lab/data/rruff_raman"
move_and_link "$REPO/qa_lab/data/rruff_zips"          "$LACIE/qa_lab/data/rruff_zips"
move_and_link "$REPO/qa_lab/data/houston2013_raw"     "$LACIE/qa_lab/data/houston2013_raw"
move_and_link "$REPO/qa_lab/data/cifar-10-batches-py" "$LACIE/qa_lab/data/cifar-10-batches-py"

echo ""
echo "=== TIER 1: Results DB ==="
move_file_and_link "$REPO/results/qa_exact_orbit_theorem_demo_2026_06_09.db" \
                   "$LACIE/results"

echo ""
echo "=== TIER 2: EEG archive (approved) ==="
move_and_link "$REPO/archive/phase_artifacts/phase2_data/eeg" \
              "$LACIE/archive/phase_artifacts/phase2_data/eeg"

echo ""
echo "=== TIER 3: .venv (symlinked — requires LaCie mounted to use Python) ==="
move_and_link "$REPO/.venv"                                  "$LACIE/venv/signal_experiments_venv"
move_and_link "$REPO/experiments/qa_ml/gptq_awq_env"        "$LACIE/experiments/qa_ml/gptq_awq_env"

echo ""
echo "=== TIER 3: ~/.cache/torch (symlinked) ==="
move_and_link "$HOME/.cache/torch"  "$LACIE/home_cache/torch"

echo ""
echo "=== TIER 3: Downloads (move only — no symlinks needed) ==="
move_only "$HOME/Downloads/Wolfram Player 14.3"    "$LACIE/home_downloads/Wolfram Player 14.3"
move_only "$HOME/Downloads/Ring35_Dataset_Txt 2"   "$LACIE/home_downloads/Ring35_Dataset_Txt_dup"
move_only "$HOME/Downloads/Claude.dmg"             "$LACIE/home_downloads/Claude.dmg"
move_only "$HOME/Downloads/Codex.dmg"              "$LACIE/home_downloads/Codex.dmg"
move_only "$HOME/Downloads/Ring35_Dataset_Txt.zip" "$LACIE/home_downloads/Ring35_Dataset_Txt.zip"

echo ""
echo "=== Verifying symlinks ==="
for link in \
    "$REPO/corpus/pepe_pose" \
    "$REPO/corpus/cmu_mocap_zhou2019" \
    "$REPO/experiments/qa_ml/pepe_ch2_rot3_rebuilt" \
    "$REPO/qa_lab/data/rruff_raman" \
    "$REPO/archive/phase_artifacts/phase2_data/eeg" \
    "$REPO/.venv"
do
    if [ -L "$link" ]; then
        echo "  ✓ $link → $(readlink "$link")"
    else
        echo "  ✗ MISSING SYMLINK: $link"
    fi
done

echo ""
echo "=== Disk usage after migration ==="
df -h /
df -h /Volumes/lacie

echo ""
echo "=== DONE ==="
