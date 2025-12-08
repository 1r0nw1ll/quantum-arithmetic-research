#!/usr/bin/env python3
"""
CHB-MIT EEG Dataset Downloader

Downloads real EEG seizure data from PhysioNet for Phase 2 validation.

Dataset: CHB-MIT Scalp EEG Database
- 24 patients with intractable seizures
- 664 hours of recordings
- 198 seizure events
- 23-channel EEG @ 256 Hz
- EDF+ format
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
DATA_DIR = Path("phase2_data/eeg/chbmit")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Subjects to download (starting with high-quality seizure data)
# chb01: 7 seizures, good quality
# chb03: 7 seizures
# chb05: 5 seizures
# chb10: 7 seizures
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb10']

# Files per subject (download subset for validation)
FILES_PER_SUBJECT = {
    'chb01': [
        'chb01-summary.txt',  # Seizure annotations
        'chb01_01.edf',      # Baseline (no seizure)
        'chb01_03.edf',      # Contains seizure at 2996s (50min)
        'chb01_04.edf',      # Contains seizure at 1467s (24min)
    ],
    'chb03': [
        'chb03-summary.txt',
        'chb03_01.edf',      # Baseline
        'chb03_02.edf',      # Contains seizure at 362s (6min)
        'chb03_03.edf',      # Contains seizure at 2162s (36min)
    ],
    'chb05': [
        'chb05-summary.txt',
        'chb05_01.edf',      # Baseline
        'chb05_06.edf',      # Contains seizure at 417s (7min)
        'chb05_13.edf',      # Contains seizure at 1086s (18min)
    ],
    'chb10': [
        'chb10-summary.txt',
        'chb10_01.edf',      # Baseline
        'chb10_12.edf',      # Contains seizure at 6578s (110min)
        'chb10_27.edf',      # Contains seizure at 2517s (42min)
    ],
}


def download_file(url: str, output_path: Path, desc: str = None) -> bool:
    """
    Download file with progress bar.

    Args:
        url: URL to download
        output_path: Where to save file
        desc: Description for progress bar

    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if file already exists
        if output_path.exists():
            print(f"  ✓ Already downloaded: {output_path.name}")
            return True

        # Make request with streaming
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        # Get file size
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=desc or output_path.name,
                leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"  ✓ Downloaded: {output_path.name} ({total_size / 1024 / 1024:.1f} MB)")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP Error {e.response.status_code}: {url}")
        return False
    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout: {url}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_subject(subject_id: str, files: list) -> dict:
    """
    Download all files for a subject.

    Args:
        subject_id: Subject identifier (e.g., 'chb01')
        files: List of files to download

    Returns:
        Dictionary with download statistics
    """
    subject_dir = DATA_DIR / subject_id
    subject_dir.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Subject: {subject_id}")
    print(f"{'='*80}")

    stats = {'total': len(files), 'success': 0, 'failed': 0, 'skipped': 0}

    for filename in files:
        url = f"{BASE_URL}{subject_id}/{filename}"
        output_path = subject_dir / filename

        if output_path.exists():
            stats['skipped'] += 1
            print(f"  ↷ Skipped (exists): {filename}")
            continue

        print(f"\n  Downloading: {filename}")
        success = download_file(url, output_path, desc=filename)

        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1

        # Rate limiting - be nice to PhysioNet servers
        time.sleep(1)

    return stats


def download_metadata():
    """Download dataset-wide metadata files."""
    print("\n" + "="*80)
    print("METADATA DOWNLOAD")
    print("="*80)

    metadata_files = [
        'RECORDS',
        'SUBJECT-INFO',
        'README',
    ]

    for filename in metadata_files:
        url = BASE_URL + filename
        output_path = DATA_DIR / filename

        if output_path.exists():
            print(f"  ↷ Skipped: {filename}")
            continue

        print(f"\n  Downloading: {filename}")
        download_file(url, output_path, desc=filename)

    print()


def verify_downloads():
    """Verify that key files were downloaded correctly."""
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    all_good = True

    for subject_id, files in FILES_PER_SUBJECT.items():
        subject_dir = DATA_DIR / subject_id
        print(f"\n{subject_id}:")

        for filename in files:
            filepath = subject_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / 1024 / 1024
                if filename.endswith('.edf'):
                    # EDF files should be ~40-50 MB (1 hour @ 256 Hz)
                    if size_mb < 5:
                        print(f"  ⚠ {filename}: {size_mb:.1f} MB (too small, may be corrupt)")
                        all_good = False
                    else:
                        print(f"  ✓ {filename}: {size_mb:.1f} MB")
                else:
                    print(f"  ✓ {filename}: {size_mb:.2f} MB")
            else:
                print(f"  ✗ {filename}: MISSING")
                all_good = False

    return all_good


def create_manifest():
    """Create manifest file listing all downloaded files."""
    manifest_path = DATA_DIR / "MANIFEST.txt"

    with open(manifest_path, 'w') as f:
        f.write("CHB-MIT EEG Dataset - Download Manifest\n")
        f.write("="*80 + "\n\n")
        f.write(f"Download location: {DATA_DIR}\n")
        f.write(f"Source: {BASE_URL}\n\n")

        f.write("Downloaded Subjects:\n")
        f.write("-"*80 + "\n")

        total_size = 0
        total_files = 0

        for subject_id in PRIORITY_SUBJECTS:
            subject_dir = DATA_DIR / subject_id
            if not subject_dir.exists():
                continue

            f.write(f"\n{subject_id}:\n")

            files = sorted(subject_dir.glob("*"))
            for filepath in files:
                size = filepath.stat().st_size
                total_size += size
                total_files += 1

                size_mb = size / 1024 / 1024
                f.write(f"  - {filepath.name} ({size_mb:.1f} MB)\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"Total: {total_files} files, {total_size / 1024 / 1024 / 1024:.2f} GB\n")

    print(f"\n✓ Manifest saved to: {manifest_path}")


def main():
    """Main download orchestrator."""
    print("="*80)
    print("CHB-MIT EEG DATASET DOWNLOADER")
    print("="*80)
    print()
    print("Dataset: CHB-MIT Scalp EEG Database")
    print("Source: PhysioNet (https://physionet.org/)")
    print(f"Destination: {DATA_DIR}")
    print()
    print("Subjects to download:")
    for subject_id in PRIORITY_SUBJECTS:
        n_files = len(FILES_PER_SUBJECT[subject_id])
        print(f"  - {subject_id}: {n_files} files")
    print()

    # Estimate download size
    # Each EDF file is ~40-50 MB, text files are small
    estimated_edf_files = sum(
        len([f for f in files if f.endswith('.edf')])
        for files in FILES_PER_SUBJECT.values()
    )
    estimated_size_gb = estimated_edf_files * 45 / 1024  # 45 MB per file avg

    print(f"Estimated download size: ~{estimated_size_gb:.1f} GB")
    print()

    # Confirm download
    try:
        response = input("Proceed with download? [y/N]: ")
        if response.lower() != 'y':
            print("\nDownload cancelled.")
            return
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        return

    # Download metadata
    download_metadata()

    # Download subjects
    all_stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0
    }

    for subject_id in PRIORITY_SUBJECTS:
        files = FILES_PER_SUBJECT[subject_id]
        stats = download_subject(subject_id, files)

        # Aggregate stats
        for key in all_stats:
            all_stats[key] += stats[key]

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"\nTotal files: {all_stats['total']}")
    print(f"  ✓ Downloaded: {all_stats['success']}")
    print(f"  ↷ Skipped (already exists): {all_stats['skipped']}")
    print(f"  ✗ Failed: {all_stats['failed']}")

    # Verify downloads
    all_good = verify_downloads()

    # Create manifest
    create_manifest()

    # Final status
    print("\n" + "="*80)
    if all_good and all_stats['failed'] == 0:
        print("✓ DOWNLOAD COMPLETE - All files verified")
    elif all_stats['failed'] > 0:
        print("⚠ DOWNLOAD COMPLETE - Some files failed (see above)")
    else:
        print("⚠ DOWNLOAD COMPLETE - Some files may be incomplete")

    print("="*80)
    print()

    # Usage instructions
    print("Next steps:")
    print("  1. Process EDF files: python process_chbmit_eeg.py")
    print("  2. Run validation: python run_phase2_validation.py")
    print()
    print(f"Data location: {DATA_DIR}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
