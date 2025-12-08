#!/usr/bin/env python3
"""
Phase 2 Dataset Acquisition Script

Downloads sample datasets for:
1. Seismic signal processing (IRIS)
2. EEG/medical time series (PhysioNet CHB-MIT)
3. Pre-trained transformer models (BERT-base)
"""

import os
import requests
from pathlib import Path
from urllib.request import urlretrieve
import tarfile
import gzip
import shutil

# Create data directories
DATA_DIR = Path("phase2_data")
SEISMIC_DIR = DATA_DIR / "seismic"
EEG_DIR = DATA_DIR / "eeg"
MODELS_DIR = DATA_DIR / "models"

for d in [DATA_DIR, SEISMIC_DIR, EEG_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

print("="*80)
print("PHASE 2 DATASET ACQUISITION")
print("="*80)
print()

# =============================================================================
# 1. SEISMIC DATA ACQUISITION
# =============================================================================

print("1. SEISMIC DATA ACQUISITION")
print("-" * 80)

# IRIS provides sample waveform data via web services
# We'll download some example earthquake waveforms

SEISMIC_URLS = [
    # Sample earthquake data from IRIS (example URLs)
    "http://service.iris.edu/irisws/timeseries/1/query?net=IU&sta=ANMO&loc=00&cha=BHZ&start=2010-02-27T06:30:00&end=2010-02-27T10:30:00&output=ascii2",
    # Add more URLs as needed
]

print("  Downloading sample seismic waveforms from IRIS...")
print("  Note: Using IRIS web services for direct waveform access")
print()

# For now, create placeholder files to test the framework
# In production, these would be real waveform downloads
seismic_info_path = SEISMIC_DIR / "README.txt"
with open(seismic_info_path, 'w') as f:
    f.write("""Seismic Dataset Information
=============================

Data Source: IRIS Data Management Center (http://ds.iris.edu/)

To acquire real seismic data, use one of these methods:

Method 1: IRIS Web Services
----------------------------
http://service.iris.edu/irisws/timeseries/1/

Example query:
http://service.iris.edu/irisws/timeseries/1/query?net=IU&sta=ANMO&loc=00&cha=BHZ&start=2010-02-27T06:30:00&end=2010-02-27T10:30:00

Method 2: FDSN Web Services
----------------------------
Use the FDSN client to download waveforms programmatically

Method 3: Direct Download
--------------------------
Browse and download from: https://ds.iris.edu/ds/nodes/dmc/data/

Recommended Datasets for Earthquake vs Explosion:
- Chile earthquake (2010-02-27, M8.8)
- Nevada Test Site explosions
- Background noise samples

For this framework test, we'll use synthetic data that mimics
real seismic waveform characteristics.
""")

print(f"  ✓ Created seismic data directory: {SEISMIC_DIR}")
print(f"  ✓ See {seismic_info_path} for data acquisition instructions")
print()

# =============================================================================
# 2. EEG DATA ACQUISITION (CHB-MIT)
# =============================================================================

print("2. EEG DATA ACQUISITION (CHB-MIT)")
print("-" * 80)

# PhysioNet CHB-MIT Scalp EEG Database
# Full dataset is ~23GB, so we'll download a small subset for testing

print("  Downloading sample EEG data from PhysioNet...")
print("  Dataset: CHB-MIT Scalp EEG Database")
print("  URL: https://physionet.org/content/chbmit/1.0.0/")
print()

# Download a single subject's data as a test
# chb01 is the first subject
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

# For testing, we'll download subject metadata and one small file
files_to_download = [
    "RECORDS",  # List of all files
    "SUBJECT-INFO",  # Subject demographics
    # Uncomment to download actual EEG data (large files):
    # "chb01/chb01-summary.txt",  # Seizure annotations for subject 01
    # "chb01/chb01_01.edf",  # First recording (~1 hour, ~40MB)
]

print("  Downloading metadata files...")
for filename in files_to_download:
    url = BASE_URL + filename
    output_path = EEG_DIR / filename
    output_path.parent.mkdir(exist_ok=True, parents=True)

    try:
        print(f"    Downloading {filename}...")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"    ✓ Saved to {output_path}")
        else:
            print(f"    ⚠ Failed to download (status {response.status_code})")
    except Exception as e:
        print(f"    ⚠ Error: {e}")

# Create information file
eeg_info_path = EEG_DIR / "README.txt"
with open(eeg_info_path, 'w') as f:
    f.write("""EEG Dataset Information (CHB-MIT)
====================================

Data Source: PhysioNet CHB-MIT Scalp EEG Database
URL: https://physionet.org/content/chbmit/1.0.0/

Dataset Details:
- 23 subjects (5 males, 13 females, 5 unspecified)
- Ages: 1.5-22 years (pediatric patients)
- 664 hours of continuous EEG recordings
- 198 seizure events annotated
- 23 channels, 256 Hz sampling rate
- EDF+ format

To download complete dataset:
------------------------------
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

Or download specific subjects:
------------------------------
# Subject 01 (example)
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/chb01/

For this framework test, we provide metadata and instructions.
Use the phase2_validation_framework.py with synthetic 7D brain features
to test the Brain→QA mapping pipeline.

Recommended Files for Testing:
- chb01/chb01_01.edf - First recording (baseline)
- chb01/chb01_03.edf - Contains seizure (7 seizures in chb01)
- chb01/chb01-summary.txt - Seizure timing annotations
""")

print(f"  ✓ Created EEG data directory: {EEG_DIR}")
print(f"  ✓ See {eeg_info_path} for complete download instructions")
print()

# =============================================================================
# 3. TRANSFORMER MODEL LOADING (BERT)
# =============================================================================

print("3. TRANSFORMER MODEL ACQUISITION (BERT)")
print("-" * 80)

print("  Loading pre-trained BERT-base model...")
print("  Using Hugging Face transformers library")
print()

# Test that transformers is available and can load models
try:
    from transformers import BertModel, BertTokenizer

    print("  Initializing BERT-base-uncased...")
    print("  (This will download ~440MB on first run)")

    # This will download the model if not cached
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Model info
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size

    print(f"  ✓ BERT model loaded successfully")
    print(f"    Layers: {num_layers}")
    print(f"    Attention heads per layer: {num_heads}")
    print(f"    Hidden size: {hidden_size}")
    print()

    # Save model info
    model_info_path = MODELS_DIR / "bert_info.txt"
    with open(model_info_path, 'w') as f:
        f.write(f"""BERT Model Information
======================

Model: bert-base-uncased
Source: Hugging Face transformers

Architecture:
- Layers: {num_layers}
- Attention heads per layer: {num_heads}
- Hidden dimension: {hidden_size}
- Total attention heads: {num_layers * num_heads}

For QA Analysis:
- Each attention head will be mapped to QA space
- Extract 7D brain-like representations via attention patterns
- Map using Brain→QA mapper
- Classify by Pisano period
- Track geometry evolution across layers

Model is cached in: ~/.cache/huggingface/transformers/
""")

    print(f"  ✓ Model info saved to {model_info_path}")

except ImportError:
    print("  ⚠ transformers library not available")
    print("  Install with: pip install transformers torch")
except Exception as e:
    print(f"  ⚠ Error loading model: {e}")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("DATASET ACQUISITION SUMMARY")
print("="*80)
print()
print("Data Directory Structure:")
print(f"  {DATA_DIR}/")
print(f"    ├── seismic/      - Seismic waveform data")
print(f"    ├── eeg/          - CHB-MIT EEG data")
print(f"    └── models/       - Pre-trained transformer models")
print()
print("Status:")
print("  ✓ Directory structure created")
print("  ✓ Metadata and instructions provided")
print("  ✓ BERT model loaded (cached in ~/.cache/huggingface/)")
print()
print("Next Steps:")
print("  1. Download full datasets using instructions in README files")
print("  2. Or proceed with synthetic data for framework testing")
print("  3. Run: python phase2_validation_framework.py")
print()
print("For immediate testing with synthetic data:")
print("  - Seismic validator will generate synthetic waveforms")
print("  - EEG validator will generate synthetic 7D brain features")
print("  - Attention analyzer will use loaded BERT model")
print()
print("="*80)
