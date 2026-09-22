"""Step 1: Load PS1 and make windows of 60 consecutive readings."""
from pathlib import Path
import numpy as np
import pandas as pd

WINDOW_SIZE = 60


def load_windows(file_path, data_format="cycles", windows_per_cycle=3):
    """Return windows and cycle IDs for safe train/validation/test splitting.

    data_format="cycles": One raw hydraulic cycle per row (6000 PS1 values in UCI PS1.txt).
    data_format="windows": Already-prepared table with the 60 readings in columns 1..60.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find {file_path}. Check DATA_FILE at the top of main.py."
        )
    if windows_per_cycle < 1:
        raise ValueError("WINDOWS_PER_CYCLE must be at least 1.")

    extension = file_path.suffix.lower()
    if extension in (".xlsx", ".xls"):
        table = pd.read_excel(file_path, header=None)
    elif extension in (".txt", ".dat"):
        # Faster than a Python regular-expression parser for large UCI PS1.txt.
        table = pd.read_csv(file_path, header=None, sep=r"\s+")
    elif extension == ".csv":
        # Determine whether the user's CSV uses commas or semicolons.
        with file_path.open(encoding="utf-8-sig") as f:
            first_line = f.readline()
        separator = ";" if first_line.count(";") > first_line.count(",") else ","
        table = pd.read_csv(file_path, header=None, sep=separator)
    else:
        raise ValueError("Use a TXT, CSV, XLS or XLSX data file.")

    table = table.apply(pd.to_numeric, errors="coerce")
    table = table.dropna(axis=0, how="all")
    rows = table.to_numpy(dtype=np.float32)

    if data_format == "windows":
        if rows.shape[1] < WINDOW_SIZE:
            raise ValueError("Windowed data must have 60 reading columns.")
        windows = rows[:, :WINDOW_SIZE]
        if not np.isfinite(windows).all():
            raise ValueError("Some prepared windows contain missing/non-numeric readings.")
        # Already-windowed files do not provide original cycle IDs.
        # Treat every row as independent unless you provide cycle IDs yourself.
        groups = np.arange(len(windows))
        return windows, groups

    if data_format != "cycles":
        raise ValueError('DATA_FORMAT must be "cycles" or "windows".')

    windows = []
    groups = []
    for cycle_id, row in enumerate(rows):
        row = row[np.isfinite(row)]  # Ignore trailing missing cells, if present.
        if len(row) < WINDOW_SIZE:
            continue

        # Pick the requested number of evenly spaced windows from EACH cycle.
        last_start = len(row) - WINDOW_SIZE
        start_positions = np.linspace(
            0, last_start, windows_per_cycle, dtype=int
        )
        for start in start_positions:
            windows.append(row[start:start + WINDOW_SIZE])
            groups.append(cycle_id)

    if not windows:
        raise ValueError("The file contains no complete 60-reading windows.")

    return np.asarray(windows, dtype=np.float32), np.asarray(groups)
