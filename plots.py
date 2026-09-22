"""Save simple plots for threshold choice and each diagnosis method."""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def safe_filename(name):
    """Example: 'Random Forest' -> 'random_forest'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_thresholds(table, chosen_threshold, figures_folder):
    figures_folder = Path(figures_folder)
    figures_folder.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for metric in ["precision", "recall", "f1", "accuracy"]:
        plt.plot(table["threshold"], table[metric], label=metric)
    plt.axvline(chosen_threshold, linestyle="--", label="chosen threshold")
    plt.xlabel("Autoencoder reconstruction-error threshold")
    plt.ylabel("Validation metric")
    plt.title("Choose detection threshold on validation data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_folder / "threshold_curves.png")
    plt.close()


def plot_comparison(table, title, path):
    """One graph for comparing all diagnosis methods on the SAME test examples."""
    ax = table.set_index("classifier")[["precision", "recall", "f1", "accuracy"]].plot.bar(
        figsize=(13, 5), ylim=(0, 1)
    )
    ax.set_title(title)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Test metric")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, name, path, labels, display_labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=labels, display_labels=display_labels,
        ax=ax, xticks_rotation=40,
        values_format="d",
    )
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_history(history, name, path):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="train loss")
    plt.plot(history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(name + " training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
