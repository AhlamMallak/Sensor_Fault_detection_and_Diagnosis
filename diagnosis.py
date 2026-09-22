"""Phase 2: train and use the seven classical classifiers + CNN + LSTM.

All classifiers learn the same four fault labels:
1 gain, 2 bias, 3 constant, 4 constant-zero.
The detector decides whether a window is healthy BEFORE diagnosis runs.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


CLASSIFIER_NAMES = [
    "LDA", "Logistic Regression", "KNN", "Decision Tree",
    "Naive Bayes", "SVM", "Random Forest", "CNN", "LSTM",
]


def make_classical_classifiers():
    """Each method is visible here, so you can modify one at a time."""
    return {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=80, random_state=42),
    }


def build_cnn():
    """A small, straightforward 1-D convolutional network."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense

    model = Sequential([
        Input(shape=(60, 1)),
        Conv1D(32, kernel_size=3, padding="same", activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm():
    """A small LSTM network for classifying 60-reading windows."""
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense

    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(32),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_all_classifiers(X_train, y_train, X_val, y_val, epochs=12,
                          selected_names=None):
    """Fit all selected methods using exactly the same train/validation groups.

    Classical classifiers receive raw 60-value rows through their own pipelines.
    CNN and LSTM receive (samples, 60, 1) scaled with a TRAIN-only scaler.
    We convert labels 1..4 to 0..3 just for the neural networks.
    """
    if selected_names is None:
        selected_names = CLASSIFIER_NAMES

    models = {}
    histories = {}
    deep_scaler = None

    print("\nTraining classical diagnosis classifiers:")
    classical = make_classical_classifiers()
    for name in selected_names:
        if name not in classical:
            continue
        print(" -", name)
        model = classical[name]
        model.fit(X_train, y_train)
        models[name] = model

    if "CNN" in selected_names or "LSTM" in selected_names:
        from tensorflow.keras.callbacks import EarlyStopping

        deep_scaler = MinMaxScaler()
        deep_scaler.fit(X_train.reshape(-1, 1))
        Xtr = deep_scaler.transform(X_train.reshape(-1, 1))
        Xva = deep_scaler.transform(X_val.reshape(-1, 1))
        Xtr = Xtr.reshape(X_train.shape)[..., None]
        Xva = Xva.reshape(X_val.shape)[..., None]

        for name in ("CNN", "LSTM"):
            if name not in selected_names:
                continue
            print(" -", name)
            model = build_cnn() if name == "CNN" else build_lstm()
            history = model.fit(
                Xtr, y_train - 1,  # 1..4 -> 0..3
                validation_data=(Xva, y_val - 1),
                epochs=epochs, batch_size=32,
                callbacks=[EarlyStopping(
                    monitor="val_loss", patience=4, restore_best_weights=True
                )],
                verbose=1
            )
            models[name] = model
            histories[name] = history.history

    return models, deep_scaler, histories


def diagnose_faults(model, windows, name, deep_scaler=None):
    """Predict the fault TYPE for windows already flagged by the detector."""
    X = np.asarray(windows, dtype=np.float32)
    if len(X) == 0:
        return np.array([], dtype=int)

    if name in ("CNN", "LSTM"):
        if deep_scaler is None:
            raise ValueError("A saved deep_scaler is needed for CNN/LSTM.")
        Xs = deep_scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        # Neural network predicts 0..3, convert back to project labels 1..4.
        return model.predict(Xs[..., None], verbose=0).argmax(axis=1) + 1

    return model.predict(X).astype(int)



def make_metrics(real_labels, predicted_labels, labels):
    """Return the same four scores for every classifier."""
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 accuracy_score)

    return {
        "precision": precision_score(
            real_labels, predicted_labels, labels=labels,
            average="macro", zero_division=0
        ),
        "recall": recall_score(
            real_labels, predicted_labels, labels=labels,
            average="macro", zero_division=0
        ),
        "f1": f1_score(
            real_labels, predicted_labels, labels=labels,
            average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(real_labels, predicted_labels),
    }
