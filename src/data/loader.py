import wfdb
import numpy as np
import os

# Path to dataset
DATA_PATH = "dataset/mitdb"
DEFAULT_SEGMENT_LENGTH = 200


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


def segment_beats(signal, r_peaks, labels=None, segment_length=DEFAULT_SEGMENT_LENGTH, channel=0):
    """
    Extract fixed-length beat segments around each R-peak.

    Args:
        signal (np.ndarray): full ECG signal of shape (samples, channels)
        r_peaks (array-like): annotation sample indices
        labels (array-like | None): optional beat labels aligned with r_peaks
        segment_length (int): total samples per beat segment
        channel (int): ECG channel to use

    Returns:
        tuple:
            beats (np.ndarray): shape (num_beats, segment_length)
            beat_labels (np.ndarray): shape (num_beats,) if labels provided
    """
    signal = np.asarray(signal, dtype=np.float32)
    r_peaks = np.asarray(r_peaks)

    if signal.ndim == 1:
        beat_signal = signal
    else:
        beat_signal = signal[:, channel]

    half_window = segment_length // 2
    extra_right = segment_length % 2

    beats = []
    beat_labels = []

    for index, peak in enumerate(r_peaks):
        start = peak - half_window
        end = peak + half_window + extra_right

        if start < 0 or end > len(beat_signal):
            continue

        beats.append(beat_signal[start:end])

        if labels is not None:
            beat_labels.append(labels[index])

    beats = np.asarray(beats, dtype=np.float32)

    if labels is None:
        return beats

    return beats, np.asarray(beat_labels)


def extract_beats(signal, annotation, window=DEFAULT_SEGMENT_LENGTH):
    """
    Backward-compatible helper for segmented beat extraction.
    """
    return segment_beats(
        signal=signal,
        r_peaks=annotation.sample,
        labels=annotation.symbol,
        segment_length=window,
        channel=0,
    )


def load_dataset(records_list=None, segment_length=DEFAULT_SEGMENT_LENGTH):
    """
    Load a segmented beat dataset from MIT-BIH records.

    Segmentation is handled here because it depends on record annotations.
    """
    if records_list is None:
        records_list = get_all_records()

    all_beats = []
    all_labels = []

    for record in records_list:
        signal, annotation = load_record(record)

        beats, labels = extract_beats(signal, annotation, window=segment_length)

        all_beats.extend(beats)
        all_labels.extend(labels)

    return np.asarray(all_beats, dtype=np.float32), np.asarray(all_labels)
