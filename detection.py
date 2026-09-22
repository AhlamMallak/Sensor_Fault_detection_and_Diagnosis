"""Step 3: LSTM autoencoder, reconstruction error, and threshold selection."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def train_autoencoder(healthy_train, healthy_validation, epochs=12):
    """Fit scaler on TRAIN only; train autoencoder using healthy windows only."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # One scaler for all raw readings, fitted only on training data.
    scaler = MinMaxScaler()
    scaler.fit(healthy_train.reshape(-1, 1))
    X_train = scaler.transform(healthy_train.reshape(-1, 1)).reshape(healthy_train.shape)
    X_val = scaler.transform(
        healthy_validation.reshape(-1, 1)
    ).reshape(healthy_validation.shape)

    # Each example is (60, 1); Dense(60) reconstructs the 60 readings.
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(32),
        Dense(60),
    ])
    model.compile(optimizer="adam", loss="mse")

    print("Training tensor shape:", X_train[..., None].shape)
    history = model.fit(
        X_train[..., None], X_train,
        validation_data=(X_val[..., None], X_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1,
    )
    return model, scaler, history


def reconstruction_error(original, reconstructed, method="mae"):
    """Compute ONE anomaly score per 60-reading window."""
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if method == "mae":
        return np.mean(np.abs(original - reconstructed), axis=1)

    if method == "correlation":
        # Correlation is undefined for perfectly constant signals.
        # Assign these an anomaly score of 1 instead of returning NaN.
        a = original - original.mean(axis=1, keepdims=True)
        b = reconstructed - reconstructed.mean(axis=1, keepdims=True)
        numerator = np.sum(a * b, axis=1)
        denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
        correlation = np.zeros(len(original))
        np.divide(numerator, denominator, out=correlation, where=denominator > 1e-12)
        return 1.0 - np.clip(correlation, -1.0, 1.0)

    raise ValueError('ERROR_METHOD must be "mae" or "correlation".')


def choose_threshold(scores, labels):
    """Try 101 thresholds and select the one with the highest validation F1."""
    scores = np.asarray(scores)
    real_fault = np.asarray(labels) != 0  # Any nonzero fault class = fault.
    rows = []

    for threshold in np.linspace(scores.min(), scores.max(), 101):
        predicted_fault = scores > threshold
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(real_fault, predicted_fault, zero_division=0),
            "recall": recall_score(real_fault, predicted_fault, zero_division=0),
            "f1": f1_score(real_fault, predicted_fault, zero_division=0),
            "accuracy": accuracy_score(real_fault, predicted_fault),
        })

    table = pd.DataFrame(rows)
    best = table.sort_values(["f1", "accuracy"], ascending=False).iloc[0]
    return float(best["threshold"]), table
