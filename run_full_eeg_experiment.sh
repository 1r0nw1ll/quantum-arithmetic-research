#!/bin/bash
# Run full EEG experiment with all 6 available seizure files
# Expected runtime: 1-2 hours
# Will generate balanced dataset with ~150 seizure segments

echo "=========================================="
echo "FULL EEG HI 2.0 EXPERIMENT"
echo "Processing all 6 seizure files from chb01"
echo "=========================================="
echo ""
echo "This will take 1-2 hours to complete."
echo "Results will be saved to:"
echo "  - eeg_hi2_0_balanced_results.json"
echo "  - eeg_hi2_0_balanced_results_visualization.png"
echo ""
echo "Current quick run (4 files) results:"
echo "  - HI 1.0 F1: 0.585"
echo "  - HI 2.0 F1: 0.667"
echo "  - Improvement: +0.081 (not statistically significant, p=0.233)"
echo "  - Test samples: 45"
echo ""
echo "Expected full run (6 files) results:"
echo "  - Test samples: ~75-80 (67% more data)"
echo "  - Better statistical power"
echo "  - More confident p-value"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."

python eeg_hi2_0_balanced_experiment.py

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "Check results at:"
echo "  - eeg_hi2_0_balanced_results.json"
echo "  - eeg_hi2_0_balanced_results_visualization.png"
echo "=========================================="
