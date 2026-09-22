"""Step 2: Create four straightforward synthetic sensor faults."""
import numpy as np

FAULT_NAMES = {
    0: "healthy",
    1: "gain",
    2: "bias",
    3: "constant",
    4: "constant_zero",
}

# Change these to match your experimental fault definitions if needed.
GAIN_FACTOR = 1.25
BIAS_VALUE = 10.0  # in the same units as the original PS1 values (bar)


def make_faults(healthy_windows):
    """Return four synthetic fault types and their numeric labels (1 to 4)."""
    healthy = np.asarray(healthy_windows, dtype=np.float32)
    gain = healthy * GAIN_FACTOR
    bias = healthy + BIAS_VALUE
    constant = np.repeat(healthy[:, :1], healthy.shape[1], axis=1)
    zero = np.zeros_like(healthy)

    X = np.vstack([gain, bias, constant, zero])
    y = np.concatenate([
        np.full(len(healthy), 1),
        np.full(len(healthy), 2),
        np.full(len(healthy), 3),
        np.full(len(healthy), 4),
    ])
    return X, y


def make_detection_examples(healthy_windows):
    """Combine healthy and faulty windows to evaluate the full pipeline."""
    faults, fault_labels = make_faults(healthy_windows)
    X = np.vstack([healthy_windows, faults])
    y = np.concatenate([np.zeros(len(healthy_windows), dtype=int), fault_labels])
    return X, y
