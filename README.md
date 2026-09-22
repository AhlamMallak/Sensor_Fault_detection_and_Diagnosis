# Hydraulic Sensor Fault Detection and Diagnosis

## Research Background

This repository presents a Python implementation of the sensor fault
detection and diagnosis framework developed in my published research:

**Paper:** Sensor and Component Fault Detection and Diagnosis for
Hydraulic Machinery Integrating LSTM Autoencoder Detector and
Diagnostic Classifiers

**Authors:** Ahlam Mallak and Madjid Fathi

**Published in:** Sensors, 2021, Volume 21, Issue 2, Article 433

**DOI:** https://doi.org/10.3390/s21020433

### Research Abstract — Summary

Hydraulic machinery operates in demanding industrial environments,
where faults affecting mechanical components and monitoring sensors
can lead to equipment failures, operational disruptions, and safety
risks.

This research introduces a two-stage fault detection and diagnosis
(FDD) framework that combines an LSTM autoencoder with supervised
machine learning and deep learning classifiers.

In the first stage, the LSTM autoencoder learns the behavior of
healthy operating conditions and reconstructs input sensor signals.
Differences between the original and reconstructed signals are used
to detect potential faults.

In the second stage, diagnostic classifiers identify the type of
fault detected by the autoencoder.

The original research investigates this framework using hydraulic
test-rig data and includes two experimental applications: sensor
fault detection and diagnosis, and hydraulic component fault
detection and diagnosis.

The study demonstrates how combining LSTM-based anomaly detection
with supervised classification can support automated condition
monitoring of industrial machinery.

## About This Implementation

This repository focuses on the sensor fault detection and diagnosis
part of the published research, using pressure sensor PS1.

The Python implementation includes:

- LSTM autoencoder for healthy-signal reconstruction and fault detection.
- Validation-based selection of the reconstruction-error threshold.
- Synthetic sensor fault injection: gain, bias, constant, and
  constant-zero faults.
- Comparison of seven classical machine learning classifiers:
  LDA, Logistic Regression, KNN, Decision Tree, Naive Bayes,
  SVM, and Random Forest.
- Two deep learning classifiers: CNN and LSTM.
- Evaluation using precision, recall, F1-score, accuracy,
  and confusion matrices.
- Jupyter notebooks for data exploration, model training,
  threshold selection, and classifier comparison.

**Implementation note:** This repository is a simplified,
modular adaptation of the sensor-FDD methodology described in
the original publication. It uses 60 consecutive PS1 readings
per window and is not an exact reproduction of every experimental
setting or result reported in the paper.

How to use: 

# PS1 fault detection and diagnosis — SIMPLE PyCharm project

**Open `main.py` and click the green Run button.**

This version keeps the simple five-step coding style. It now includes all
nine diagnosis methods, **without** a `src/` package or command-line arguments.

```text
PS1_FDD_SIMPLE_ALL_CLASSIFIERS/
├── main.py                     ← Run TRAIN or PREDICT here
├── data.py                     ← Step 1: read and window PS1
├── faults.py                   ← Step 2: create four sensor fault types
├── detection.py                ← Step 3: LSTM + LEARNED threshold
├── diagnosis.py                ← Step 4: train nine classifiers
├── plots.py                    ← Step 5: save research graphs
├── notebooks/
│   ├── 01_data_and_windows.ipynb
│   ├── 02_fault_detection.ipynb
│   └── 03_fault_diagnosis.ipynb
├── data/
│   └── PS1_demo.csv            ← GENERATED demo, NOT actual UCI data
├── saved_models/               ← Models saved after training
├── results/
│   └── figures/                ← Scores, plots, confusion matrices
├── requirements.txt
└── README.md
```

## First run in PyCharm

1. Unzip to a **NEW** folder. Do not merge with earlier project versions.
2. Open the folder in PyCharm and choose a Python 3.10 or 3.11 interpreter.
3. In PyCharm's Terminal run:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Open `main.py`. Leave the demonstration settings unchanged:
   `MODE = "train"` and `DATA_FILE = "data/PS1_demo.csv"`.
5. Click the green Run button. The project runs the full detector +
   nine-classifier diagnosis comparison and saves the models and figures.
6. To reuse trained models on new readings, change `MODE = "predict"` and
   `DATA_FILE` to the new data file, then run `main.py` again. Set
   `PREDICT_CLASSIFIER` to the classifier you want to use.

**Note:** TensorFlow must be installed and import correctly to train the
autoencoder, CNN, and LSTM. Running all nine classifiers is more demanding
than the prior Random-Forest-only example. For a quick first test, try
`EPOCHS = 5` with the included demonstration data.

## What is the learned detection threshold?

1. Train the LSTM autoencoder **only on healthy training cycles**.
2. Create separate **healthy and synthetically faulty validation cycles**.
3. Reconstruct those validation windows and calculate **one error score
   per window** (MAE by default, or correlation distance).
4. In `detection.py`, try 101 candidate error thresholds. For each,
   calculate fault-detection precision, recall, F1 and accuracy.
5. Select the threshold that gives the **highest validation F1**, with
   validation accuracy as the tie-breaker.
6. Save the selected number and metric name in
   `saved_models/detector_settings.json`. When predicting on new data,
   `main.py` LOADS this number. It does NOT refit the threshold.

A learned/calibrated threshold does **not** mean the LSTM has a special
"threshold neuron". The LSTM learns to reconstruct healthy signals;
the separate validation procedure then calibrates the cutoff.

The threshold metrics/curves are saved in `results/threshold_curves.csv`
and `results/figures/threshold_curves.png`. Notebook 02 shows the
calculation and graph step by step.

The automatic maximum-F1 rule is a *simplification* of your earlier
visual metric-curve analysis, not an exact reconstruction of that research
threshold decision. The synthetic faults, their severity, and the number
of healthy/faulty validation examples affect the learned cutoff.

## Which classifiers are included?

All nine classifiers are listed clearly in `diagnosis.py`:

1. Linear Discriminant Analysis (LDA)
2. Logistic Regression
3. K-Nearest Neighbors (KNN)
4. Decision Tree (CART)
5. Gaussian Naive Bayes
6. Support Vector Machine (SVM)
7. Random Forest
8. Convolutional Neural Network (CNN)
9. Long Short-Term Memory (LSTM)

They all diagnose the same four labels: **gain, bias, constant,
constant-zero**. They are given the same train/validation/test cycle
splits. The first seven take the 60 raw readings as features; CNN and
LSTM use a `(samples, 60, 1)` tensor. Neural classifiers get their own
scaler fitted **on training data only**.

`main.py` reports TWO different test comparisons:

- **Diagnosis-only:** Classify the four fault types on test windows whose
  fault status is known. This tests the classifier independently.
- **Full system:** Run the learned LSTM detector first; classify **only
  windows it detects as faulty**. This includes any detection errors.

All comparisons use held-out **TEST** cycles that were never used to
fit the models or the threshold. Precision, recall and F1 for the
diagnosis methods are **macro averages** (the unweighted mean across
the four classes, or five classes for the full system).

## Where do I find the graphs?

After `MODE = "train"` succeeds:

```text
saved_models/
    detector.keras
    detector_scaler.joblib
    detector_settings.json
    classifier_lda.joblib
    classifier_logistic_regression.joblib
    ... [all seven classical classifiers]
    classifier_cnn.keras
    classifier_lstm.keras
    deep_scaler.joblib

results/
    threshold_curves.csv
    detector_test_metrics.csv
    diagnosis_comparison.csv
    full_system_comparison.csv
    test_predictions_each_method.csv
    figures/
        threshold_curves.png
        detector_training.png
        compare_diagnosis.png
        compare_full_system.png
        confusion_diagnosis_lda.png
        confusion_full_system_lda.png
        ... [one diagnosis and one full-system matrix for EVERY classifier]
        training_cnn.png
        training_lstm.png
```

Notebook 03 also draws each method's confusion matrix and the common
precision/recall/F1/accuracy bar chart. Graphs are saved even when the
PyCharm plot pane is not open.

## Use real PS1 measurements

For the original UCI `PS1.txt`, copy the file into `data/`:

```python
DATA_FILE = "data/PS1.txt"
DATA_FORMAT = "cycles"
```

The UCI PS1.txt rows correspond to separate **60-second cycles**. The
script takes `WINDOWS_PER_CYCLE` windows from each cycle and **keeps
all windows from the same original cycle in one train/validation/test
partition**.

For your existing Excel file with 60 PS1 readings per row, use:

```python
DATA_FILE = "data/your_windows.xlsx"
DATA_FORMAT = "windows"
```

Only the **first 60 columns** are treated as the window. If your Excel
rows have original cycle IDs, the simple loader **cannot automatically
recover these IDs**. Set up proper cycle grouping before using its
metrics for scientific claims.

**Crucial:** The full raw UCI file also contains different hydraulic
COMPONENT operating conditions. Select genuine healthy operating cycles
for the autoencoder training rather than assuming every PS1 cycle is
component-healthy. The included demo has only simulated healthy cycles.

### 60 readings versus 60 seconds

This simplified code uses **60 consecutive raw PS1 readings**, which
correspond to **0.6 seconds when PS1 is sampled at 100 Hz**.
The original publication's *60-second* windows are **not** identical.
Synthetic fault constants in `faults.py` are explicit examples; update
them for an exact reproduction of your original experimental design.

## Model selection and scientific limitations

The generated `PS1_demo.csv` contains simulated pressure-like signals,
not actual research observations. Synthetic fault scores — even perfect
ones — are not evidence of real-world sensor-fault accuracy.

Do not automatically choose the "best" classifier from the final TEST
graphs and reuse its reported test accuracy as an independent assessment
of a model selected on those same graphs. For model selection, compare
on a validation set, then assess once on a separate held-out test set.
