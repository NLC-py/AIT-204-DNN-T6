"""
backend/data.py
Data loading and preprocessing for ANN, CNN, and RNN tasks.
All datasets come from scikit-learn or are synthetically generated.
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# ANN  –  Tabular classification (scikit-learn)
# ─────────────────────────────────────────────

DATASET_INFO = {
    "Iris":          {"loader": datasets.load_iris,          "classes": 3},
    "Breast Cancer": {"loader": datasets.load_breast_cancer, "classes": 2},
    "Wine":          {"loader": datasets.load_wine,          "classes": 3},
    "Digits":        {"loader": datasets.load_digits,        "classes": 10},
}


def load_ann_data(dataset_name: str, test_size: float = 0.2):
    """
    Returns
    -------
    X_train, X_test  : float32 arrays, StandardScaler-normalised
    y_train, y_test  : one-hot encoded int arrays
    n_classes        : int
    feature_names    : list[str] | None
    class_names      : list[str] | None
    """
    from tensorflow.keras.utils import to_categorical  # lazy import

    info = DATASET_INFO[dataset_name]
    data = info["loader"]()
    X, y = data.data.astype(np.float32), data.target

    n_classes = len(np.unique(y))
    feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None
    class_names   = list(data.target_names)  if hasattr(data, "target_names")  else None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    y_tr_cat = to_categorical(y_tr, n_classes)
    y_te_cat = to_categorical(y_te, n_classes)

    meta = {
        "n_samples":    len(X),
        "n_features":   X.shape[1],
        "n_classes":    n_classes,
        "feature_names": feature_names,
        "class_names":  class_names,
        "train_size":   len(X_tr),
        "test_size":    len(X_te),
    }
    return X_tr, X_te, y_tr_cat, y_te_cat, n_classes, meta


# ─────────────────────────────────────────────
# CNN  –  Image classification (sklearn Digits)
# ─────────────────────────────────────────────

def load_cnn_data(test_size: float = 0.2):
    """
    sklearn Digits dataset: 1797 samples of 8×8 greyscale digit images.

    Returns
    -------
    X_train, X_test  : shape (N, 8, 8, 1), float32 in [0, 1]
    y_train, y_test  : one-hot encoded, 10 classes
    meta             : dict with dataset statistics
    """
    from tensorflow.keras.utils import to_categorical

    digits = datasets.load_digits()
    X = (digits.data / 16.0).astype(np.float32).reshape(-1, 8, 8, 1)
    y = digits.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    meta = {
        "n_samples":  len(X),
        "img_shape":  "(8, 8, 1)",
        "n_classes":  10,
        "train_size": len(X_tr),
        "test_size":  len(X_te),
    }
    return X_tr, X_te, to_categorical(y_tr, 10), to_categorical(y_te, 10), meta


# ─────────────────────────────────────────────
# RNN  –  Time-series regression (synthetic)
# ─────────────────────────────────────────────

def load_rnn_data(seq_length: int = 40, n_samples: int = 1200,
                  noise_level: float = 0.08):
    """
    Synthetic multi-frequency sine wave.
    Task: predict the NEXT value given a window of `seq_length` past values.

    Returns
    -------
    X_train, X_test  : shape (N, seq_length, 1), float32
    y_train, y_test  : shape (N,),  float32
    t_full, series   : full time axis and signal (for plotting)
    meta             : dict with statistics
    """
    rng = np.random.default_rng(42)
    total = n_samples + seq_length + 50          # a bit extra for the plot
    t = np.linspace(0, 8 * np.pi, total)
    series = (
        np.sin(t)
        + 0.5 * np.sin(2.5 * t)
        + 0.3 * np.sin(5.0 * t)
        + noise_level * rng.standard_normal(total)
    ).astype(np.float32)

    X, y = [], []
    for i in range(n_samples):
        X.append(series[i : i + seq_length])
        y.append(series[i + seq_length])

    X = np.array(X, dtype=np.float32).reshape(-1, seq_length, 1)
    y = np.array(y, dtype=np.float32)

    split = int(0.8 * n_samples)
    meta = {
        "n_samples":   n_samples,
        "seq_length":  seq_length,
        "train_size":  split,
        "test_size":   n_samples - split,
        "noise_level": noise_level,
        "task":        "Next-step regression",
    }
    return X[:split], X[split:], y[:split], y[split:], t, series, meta
