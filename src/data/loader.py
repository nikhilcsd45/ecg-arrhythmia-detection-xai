import wfdb
import numpy as np
import os

# Path to dataset
DATA_PATH = "dataset/mitdb"


def get_all_records():
    """
    Automatically get all record names from dataset folder
    """
    files = os.listdir(DATA_PATH)

    record_names = set()

    for file in files:
        if file.endswith(".dat"):
            record_name = file.split(".")[0]
            record_names.add(record_name)

    return sorted(list(record_names))


def load_record(record_name):
    """
    Load ECG signal + annotations from MIT-BIH dataset
    """
    record_path = os.path.join(DATA_PATH, record_name)

    # Load signal
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # (samples, channels)

    # Load annotations
    annotation = wfdb.rdann(record_path, 'atr')

    return signal, annotation


def extract_beats(signal, annotation, window=180):
    """
    Extract heartbeat segments around R-peaks
    """
    beats = []
    labels = []

    r_peaks = annotation.sample
    symbols = annotation.symbol

    for i in range(len(r_peaks)):
        peak = r_peaks[i]

        start = peak - window // 2
        end = peak + window // 2

        if start < 0 or end >= len(signal):
            continue

        beat = signal[start:end, 0]  # using first channel

        beats.append(beat)
        labels.append(symbols[i])

    return np.array(beats), np.array(labels)


def load_dataset(records_list=None):
    """
    Load dataset automatically if no records provided
    """
    if records_list is None:
        records_list = get_all_records()

    all_beats = []
    all_labels = []

    for record in records_list:
        signal, annotation = load_record(record)

        beats, labels = extract_beats(signal, annotation)

        all_beats.extend(beats)
        all_labels.extend(labels)

    return np.array(all_beats), np.array(all_labels)