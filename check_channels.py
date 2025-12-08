#!/usr/bin/env python3
"""Check what channels are actually in the EDF file"""
import pyedflib
from pathlib import Path

edf_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")
f = pyedflib.EdfReader(str(edf_file))

print("="*80)
print(f"Channels in {edf_file.name}:")
print("="*80)

for i in range(f.signals_in_file):
    label = f.getLabel(i)
    print(f"{i:2d}: {label}")

f.close()

print("\n" + "="*80)
print("Expected channel names in EEGBrainFeatureExtractor:")
print("="*80)

CHANNEL_NETWORK_MAP = {
    'VIS': ['O1', 'O2', 'Oz'],
    'SMN': ['C3', 'C4', 'Cz'],
    'DAN': ['P3', 'P4', 'Pz'],
    'VAN': ['T3', 'T4', 'T5', 'T6'],
    'FPN': ['F3', 'F4', 'Fz'],
    'DMN': ['Fp1', 'Fp2'],
    'LIM': ['F7', 'F8']
}

for network, channels in CHANNEL_NETWORK_MAP.items():
    print(f"{network}: {', '.join(channels)}")

print("\n" + "="*80)
print("DIAGNOSIS: Do channel names match?")
print("="*80)
print("\nThe feature extractor is looking for standard 10-20 names (O1, C3, etc.)")
print("but CHB-MIT uses BIPOLAR MONTAGES (FP1-F7, F7-T7, etc.)")
print("\nThat's why all features are zero - NO channels match!")
