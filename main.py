"""
RUN THIS FILE in PyCharm.

TRAIN:   Learn the LSTM autoencoder, learn its detection threshold from
         validation data, train all nine diagnosis classifiers, save models
         and graphs, then evaluate once on held-out test cycles.

PREDICT: Load the saved detector, threshold and diagnosis classifier.
         Do not train again.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import load_windows
from faults import make_faults, make_detection_examples, FAULT_NAMES
from detection import train_autoencoder, reconstruction_error, choose_threshold
from diagnosis import (
    CLASSIFIER_NAMES, train_all_classifiers, diagnose_faults, make_metrics,
)
from plots import (plot_thresholds, plot_comparison, plot_confusion_matrix,
                   plot_training_history, safe_filename)


# ==================== YOUR SETTINGS ====================
MODE = "train"                   # "train" or "predict"
DATA_FILE = "data/PS1_demo.csv"   # Replace with your healthy PS1 file.
DATA_FORMAT = "cycles"           # "cycles" or "windows"
WINDOWS_PER_CYCLE = 3
EPOCHS = 12
ERROR_METHOD = "mae"             # "mae" or "correlation"
CLASSIFIERS = CLASSIFIER_NAMES   # All nine methods by default.
PREDICT_CLASSIFIER = "Random Forest"  # Or LDA, CNN, LSTM, etc.
# =======================================================

HERE = Path(__file__).resolve().parent
MODELS = HERE / "saved_models"
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"


def train():
    MODELS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Load healthy windows (each has exactly 60 consecutive PS1 readings).
    print("\nSTEP 1: Load PS1 windows")
    windows, cycle_ids = load_windows(
        HERE / DATA_FILE, DATA_FORMAT, WINDOWS_PER_CYCLE
    )
    print("Windows:", windows.shape, " | Independent cycles/rows:", len(np.unique(cycle_ids)))

    # 2. Split BEFORE fault injection. All windows from one cycle stay together.
    print("\nSTEP 2: Separate training, validation and final test cycles")
    all_cycles = np.unique(cycle_ids)
    train_cycles, other_cycles = train_test_split(
        all_cycles, test_size=0.30, random_state=42
    )
    validation_cycles, test_cycles = train_test_split(
        other_cycles, test_size=0.50, random_state=42
    )
    healthy_train = windows[np.isin(cycle_ids, train_cycles)]
    healthy_val = windows[np.isin(cycle_ids, validation_cycles)]
    healthy_test = windows[np.isin(cycle_ids, test_cycles)]
    print("Healthy train / validation / test:",
          len(healthy_train), len(healthy_val), len(healthy_test))

    # 3. Train the detector only on healthy TRAIN windows.
    print("\nSTEP 3: Train the healthy LSTM autoencoder")
    detector, detector_scaler, detector_history = train_autoencoder(
        healthy_train, healthy_val, epochs=EPOCHS
    )
    detector.save(MODELS / "detector.keras")
    joblib.dump(detector_scaler, MODELS / "detector_scaler.joblib")
    plot_training_history(detector_history.history, "Detector",
                          FIGURES / "detector_training.png")

    # 4. LEARN the threshold using labeled VALIDATION examples.
    #    The final TEST examples are not used here.
    print("\nSTEP 4: LEARN the fault-detection threshold")
    val_windows, val_labels = make_detection_examples(healthy_val)
    val_scaled = detector_scaler.transform(
        val_windows.reshape(-1, 1)
    ).reshape(val_windows.shape)
    val_reconstruction = detector.predict(val_scaled[..., None], verbose=0)
    val_scores = reconstruction_error(
        val_scaled, val_reconstruction, method=ERROR_METHOD
    )
    threshold, threshold_table = choose_threshold(val_scores, val_labels)
    print("Selected threshold:", round(threshold, 6),
          "(highest F1 on validation examples)")
    threshold_table.to_csv(RESULTS / "threshold_curves.csv", index=False)
    plot_thresholds(threshold_table, threshold, FIGURES)
    (MODELS / "detector_settings.json").write_text(
        json.dumps({"threshold": float(threshold),
                    "error_method": ERROR_METHOD}, indent=2)
    )

    # 5. Make synthetic faults separately for training, validation and test.
    print("\nSTEP 5: Prepare fault classes for diagnosis")
    train_faults, train_fault_labels = make_faults(healthy_train)
    val_faults, val_fault_labels = make_faults(healthy_val)
    test_faults, test_fault_labels = make_faults(healthy_test)

    # 6. Train all diagnosis classifiers on the SAME training windows.
    print("\nSTEP 6: Train ALL diagnosis methods")
    classifiers, deep_scaler, deep_histories = train_all_classifiers(
        train_faults, train_fault_labels,
        val_faults, val_fault_labels,
        epochs=EPOCHS, selected_names=CLASSIFIERS
    )

    for name, classifier in classifiers.items():
        file_name = safe_filename(name)
        if name in ("CNN", "LSTM"):
            classifier.save(MODELS / f"classifier_{file_name}.keras")
            plot_training_history(deep_histories[name], name,
                                  FIGURES / f"training_{file_name}.png")
        else:
            joblib.dump(classifier, MODELS / f"classifier_{file_name}.joblib")

    if deep_scaler is not None:
        joblib.dump(deep_scaler, MODELS / "deep_scaler.joblib")

    # 7. Use the trained detector ONCE on untouched TEST-cycle examples.
    #    The SAME detection decisions apply to every diagnosis classifier.
    print("\nSTEP 7: Evaluate the detector on held-out TEST cycles")
    test_windows, true_labels = make_detection_examples(healthy_test)
    test_scaled = detector_scaler.transform(
        test_windows.reshape(-1, 1)
    ).reshape(test_windows.shape)
    test_reconstruction = detector.predict(test_scaled[..., None], verbose=0)
    test_scores = reconstruction_error(
        test_scaled, test_reconstruction, method=ERROR_METHOD
    )
    detected = test_scores > threshold

    detection_scores = make_metrics(true_labels != 0, detected, labels=[False, True])
    pd.DataFrame([detection_scores]).to_csv(
        RESULTS / "detector_test_metrics.csv", index=False
    )
    print("Detector TEST F1:", round(detection_scores["f1"], 4),
          "| threshold:", round(threshold, 6))

    # 8. Score each classifier separately AND score the complete 2-stage system.
    print("\nSTEP 8: Compare classifiers and save plots for EACH method")
    diagnosis_rows = []
    full_system_rows = []

    predictions = pd.DataFrame({
        "true_label": true_labels,
        "detector_score": test_scores,
        "fault_detected": detected,
    })

    for name, classifier in classifiers.items():
        file_name = safe_filename(name)

        # A. Diagnosis-only evaluation: give each method the TRUE test faults.
        #    Measures how well each classifier recognizes the four fault types.
        diagnosis_prediction = diagnose_faults(
            classifier, test_faults, name, deep_scaler
        )
        diagnosis_rows.append({
            "classifier": name,
            **make_metrics(test_fault_labels, diagnosis_prediction, [1, 2, 3, 4]),
        })
        plot_confusion_matrix(
            test_fault_labels, diagnosis_prediction,
            name + " — fault diagnosis only",
            FIGURES / f"confusion_diagnosis_{file_name}.png",
            [1, 2, 3, 4], [FAULT_NAMES[i] for i in range(1, 5)],
        )

        # B. Real 2-stage logic: healthy by default; diagnose DETECTED faults only.
        #    Includes missed faults and false alarms made by the LSTM detector.
        full_prediction = np.zeros(len(test_windows), dtype=int)
        if detected.any():
            full_prediction[detected] = diagnose_faults(
                classifier, test_windows[detected], name, deep_scaler
            )
        full_system_rows.append({
            "classifier": name,
            **make_metrics(true_labels, full_prediction, [0, 1, 2, 3, 4]),
        })
        predictions["prediction_" + file_name] = full_prediction
        plot_confusion_matrix(
            true_labels, full_prediction, name + " — complete 2-stage FDD",
            FIGURES / f"confusion_full_system_{file_name}.png",
            [0, 1, 2, 3, 4], [FAULT_NAMES[i] for i in range(5)],
        )
        print(f"  {name:20s} | diagnosis accuracy: "
              f"{diagnosis_rows[-1]['accuracy']:.3f} | complete FDD accuracy: "
              f"{full_system_rows[-1]['accuracy']:.3f}")

    # These CSVs are perfect for your later research/Streamlit comparison plots.
    diagnosis_table = pd.DataFrame(diagnosis_rows)
    full_system_table = pd.DataFrame(full_system_rows)
    diagnosis_table.to_csv(RESULTS / "diagnosis_comparison.csv", index=False)
    full_system_table.to_csv(RESULTS / "full_system_comparison.csv", index=False)
    predictions.to_csv(RESULTS / "test_predictions_each_method.csv", index=False)

    plot_comparison(diagnosis_table, "Diagnosis of true fault windows",
                    FIGURES / "compare_diagnosis.png")
    plot_comparison(full_system_table, "Complete detector + diagnosis system",
                    FIGURES / "compare_full_system.png")

    print("\nFINISHED. Saved models in saved_models/ and graphs in results/figures/.")
    print("Threshold table: results/threshold_curves.csv")
    print("Classifier comparison: results/diagnosis_comparison.csv")
    print("Full-system comparison: results/full_system_comparison.csv")


def predict():
    print("\nSTEP 1: Load the saved detector and threshold")
    from tensorflow.keras.models import load_model

    if PREDICT_CLASSIFIER not in CLASSIFIER_NAMES:
        raise ValueError("Choose a name from CLASSIFIER_NAMES in diagnosis.py.")

    detector = load_model(MODELS / "detector.keras")
    detector_scaler = joblib.load(MODELS / "detector_scaler.joblib")
    settings = json.loads((MODELS / "detector_settings.json").read_text())

    print("\nSTEP 2: Load new PS1 data")
    windows, cycle_ids = load_windows(
        HERE / DATA_FILE, DATA_FORMAT, WINDOWS_PER_CYCLE
    )

    print("\nSTEP 3: Detect faults using the LEARNED threshold")
    scaled = detector_scaler.transform(
        windows.reshape(-1, 1)
    ).reshape(windows.shape)
    reconstructed = detector.predict(scaled[..., None], verbose=0)
    scores = reconstruction_error(
        scaled, reconstructed, method=settings["error_method"]
    )
    detected = scores > settings["threshold"]
    print("Saved threshold:", settings["threshold"],
          "| Detected faults:", int(detected.sum()))

    print("\nSTEP 4: Classify only the faulty windows with", PREDICT_CLASSIFIER)
    classifier_file = safe_filename(PREDICT_CLASSIFIER)
    if PREDICT_CLASSIFIER in ("CNN", "LSTM"):
        classifier = load_model(MODELS / f"classifier_{classifier_file}.keras")
        deep_scaler = joblib.load(MODELS / "deep_scaler.joblib")
    else:
        classifier = joblib.load(MODELS / f"classifier_{classifier_file}.joblib")
        deep_scaler = None

    predicted_labels = np.zeros(len(windows), dtype=int)
    if detected.any():
        predicted_labels[detected] = diagnose_faults(
            classifier, windows[detected], PREDICT_CLASSIFIER, deep_scaler
        )

    print("\nSTEP 5: Save results")
    RESULTS.mkdir(exist_ok=True)
    output = pd.DataFrame({
        "cycle_or_row": cycle_ids,
        "window_number": np.arange(len(windows)),
        "anomaly_score": scores,
        "fault_detected": detected,
        "diagnosis": [FAULT_NAMES[i] for i in predicted_labels],
        "diagnosis_method": PREDICT_CLASSIFIER,
    })
    output.to_csv(RESULTS / "new_predictions.csv", index=False)
    print(output.head(12).to_string(index=False))
    print("Saved: results/new_predictions.csv")


if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "predict":
        predict()
    else:
        raise ValueError('MODE must be "train" or "predict".')
