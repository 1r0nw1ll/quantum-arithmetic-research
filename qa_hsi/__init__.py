"""
qa_hsi — Zero-parameter spatial feature extraction for hyperspectral image classification.

Maps each pixel to QA coordinates via Manhattan distances to image anchors:
  pixel (row, col) → dist_L, dist_R → b = dist_L+1, e = dist_R+1 → QA packet

Concatenate the resulting feature matrix with spectral features and pass to
any classifier (Random Forest, SVM, MLP). No convolution, no learned weights.

Validated gains (permuted-control passed, seeds 0-4, train_frac=0.10):
  Indian Pines : +0.208 OA over spectral-only (0.748 → 0.955)
  Pavia Univ.  : +0.108 OA over spectral-only (0.885 → 0.993)

Usage
-----
    from qa_hsi import QAHSITransformer
    import numpy as np

    tr = QAHSITransformer()
    tr.fit(train_rows, train_cols, train_labels, image_shape=(H, W))

    X_qa    = tr.transform(all_rows, all_cols)
    X_train = np.column_stack([spectral_train, X_qa[train_idx]])
    X_test  = np.column_stack([spectral_test,  X_qa[test_idx]])
"""

from .transformer import QAHSITransformer

__all__ = ["QAHSITransformer"]
__version__ = "0.1.0"
