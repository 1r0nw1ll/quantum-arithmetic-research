#!/usr/bin/env python3
"""
Inspect multi-modal remote sensing data (HSI + LIDAR + MS)
"""

import scipy.io as sio
import numpy as np

print("="*70)
print("MULTI-MODAL DATA INSPECTION")
print("="*70)
print()

# Load all modalities
hsi = sio.loadmat('multimodal_data/HSI_Tr.mat')
lidar = sio.loadmat('multimodal_data/LIDAR_Tr.mat')
ms = sio.loadmat('multimodal_data/MS_Tr.mat')
labels = sio.loadmat('multimodal_data/TrLabel.mat')

# Extract actual arrays (skip MATLAB metadata)
hsi_keys = [k for k in hsi.keys() if not k.startswith('__')]
lidar_keys = [k for k in lidar.keys() if not k.startswith('__')]
ms_keys = [k for k in ms.keys() if not k.startswith('__')]
label_keys = [k for k in labels.keys() if not k.startswith('__')]

print("Keys found:")
print(f"  HSI: {hsi_keys}")
print(f"  LIDAR: {lidar_keys}")
print(f"  MS: {ms_keys}")
print(f"  Labels: {label_keys}")
print()

# Get data arrays
hsi_data = hsi[hsi_keys[0]]
lidar_data = lidar[lidar_keys[0]]
ms_data = ms[ms_keys[0]]
label_data = labels[label_keys[0]]

print("HSI (Hyperspectral Imaging):")
print(f"  Shape: {hsi_data.shape}")
print(f"  Dtype: {hsi_data.dtype}")
print(f"  Range: [{hsi_data.min():.2f}, {hsi_data.max():.2f}]")
print()

print("LIDAR (Elevation):")
print(f"  Shape: {lidar_data.shape}")
print(f"  Dtype: {lidar_data.dtype}")
print(f"  Range: [{lidar_data.min():.2f}, {lidar_data.max():.2f}]")
print()

print("MS (Multispectral):")
print(f"  Shape: {ms_data.shape}")
print(f"  Dtype: {ms_data.dtype}")
print(f"  Range: [{ms_data.min():.2f}, {ms_data.max():.2f}]")
print()

print("Labels:")
print(f"  Shape: {label_data.shape}")
print(f"  Dtype: {label_data.dtype}")
print(f"  Unique classes: {np.unique(label_data)}")
print(f"  Class distribution:")
for cls in np.unique(label_data):
    count = np.sum(label_data == cls)
    print(f"    Class {cls}: {count} samples")
print()

# Check if spatial dimensions match
print("Spatial Alignment Check:")
print(f"  HSI: {hsi_data.shape}")
print(f"  LIDAR: {lidar_data.shape}")
print(f"  MS: {ms_data.shape}")
print(f"  Labels: {label_data.shape}")
print()

# Determine data format
if len(hsi_data.shape) == 3:
    print("Data format: SPATIAL (H × W × Bands)")
    H, W, B_hsi = hsi_data.shape
    print(f"  Spatial dimensions: {H} × {W}")
    print(f"  HSI bands: {B_hsi}")
elif len(hsi_data.shape) == 2:
    print("Data format: PIXELS (N_samples × Bands)")
    N, B_hsi = hsi_data.shape
    print(f"  Total samples: {N}")
    print(f"  HSI bands: {B_hsi}")
print()

print("="*70)
print("SUMMARY")
print("="*70)
print()
print("✓ Multi-modal dataset loaded successfully")
print(f"✓ HSI: {hsi_data.shape} - spectral information")
print(f"✓ LIDAR: {lidar_data.shape} - elevation/geometry")
print(f"✓ MS: {ms_data.shape} - broader spectral bands")
print(f"✓ Labels: {label_data.shape} - ground truth")
print()
print("Next step: Create fusion pipeline combining all three modalities")
print()
