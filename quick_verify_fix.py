#!/usr/bin/env python3
"""Quick verification that fixed feature extractor works on real CHB-MIT data"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from process_real_chbmit_data import RealEEGProcessor

print("="*80)
print("QUICK VERIFICATION: Fixed Feature Extractor on Real CHB-MIT Data")
print("="*80)
print()

processor = RealEEGProcessor()

# Load just the first file
baseline_file = Path("phase2_data/eeg/chbmit/chb01/chb01_01.edf")

print(f"Loading: {baseline_file}")
data = processor.load_edf_file(baseline_file)

print(f"\nActual channel names in file:")
for i, ch in enumerate(data['channel_names'][:10]):
    print(f"  {i:2d}: {ch}")
print(f"  ... (showing first 10 of {len(data['channel_names'])} channels)")

# Extract features from just first 10 seconds
print(f"\nExtracting features from first 10 seconds...")
segment_samples = int(10.0 * processor.sampling_rate)
first_segment = data['signals'][:, :segment_samples]

channels_data = {}
for i, ch_name in enumerate(data['channel_names']):
    channels_data[ch_name] = first_segment[i, :]

# Test channel mapping
print(f"\nTesting channel → network mapping:")
for ch in list(channels_data.keys())[:5]:
    networks = processor.extractor.map_channel_to_networks(ch)
    print(f"  {ch:15s} → {networks}")

# Extract features
features = processor.extractor.extract_network_features(channels_data)

print(f"\n" + "="*80)
print("RESULT")
print("="*80)
print(f"7D Brain Network Features:")
network_names = ['VIS', 'SMN', 'DAN', 'VAN', 'FPN', 'DMN', 'LIM']
for i, name in enumerate(network_names):
    print(f"  {name}: {features[i]:.6f}")

print(f"\nFeature vector: {features}")
print(f"Feature norm: {np.linalg.norm(features):.6f}")
print(f"All zeros? {np.all(features == 0)}")
print(f"Any non-zero? {np.any(features != 0)}")

if np.all(features == 0):
    print("\n❌ FAILED: Features are still all zeros!")
elif np.any(features != 0):
    print("\n✅ SUCCESS: Features are no longer all zeros!")
    print("   Fixed extractor is working correctly on real CHB-MIT data!")
else:
    print("\n⚠️  UNEXPECTED: Unclear result")

print("="*80)
