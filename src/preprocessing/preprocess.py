import numpy as np


def normalize(signal):
    """
    Normalize a 1D ECG signal to zero mean and unit variance.

    Args:
        signal (array-like): 1D ECG signal

    Returns:
        np.ndarray: normalized 1D signal
    """
    signal = np.asarray(signal, dtype=np.float32)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def preprocess_batch(X):
    """
    Apply normalization to each ECG signal/beat in a batch.

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: normalized batch of shape (batch, length)
    """
    X = np.asarray(X, dtype=np.float32)
    return np.array([normalize(sig) for sig in X], dtype=np.float32)


def reshape_for_cnn(X):
    """
    Reshape data for PyTorch Conv1D.

    PyTorch Conv1D expects:
        (batch, channels, length)

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, 1, length)
    """
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], 1, X.shape[1])


def reshape_for_lstm(X):
    """
    Reshape data for LSTM.

    Common expected shape:
        (batch, sequence_length, features)

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_for_transformer(X):
    """
    Reshape data for Transformer.

    Common expected shape:
        (batch, sequence_length, features)

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], 1)


def reshape_for_cnn_transformer(X):
    """
    Reshape data for CNN + Transformer hybrid.

    This usually depends on your architecture. For now we keep:
        (batch, sequence_length, features)

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = np.asarray(X, dtype=np.float32)
    return X.reshape(X.shape[0], X.shape[1], 1)


def prepare_for_cnn(X):
    """
    Full pipeline for CNN:
        normalize batch -> reshape for PyTorch CNN

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, 1, length)
    """
    X = preprocess_batch(X)
    X = reshape_for_cnn(X)
    return X


def prepare_for_lstm(X):
    """
    Full pipeline for LSTM:
        normalize batch -> reshape for LSTM

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = preprocess_batch(X)
    X = reshape_for_lstm(X)
    return X


def prepare_for_transformer(X):
    """
    Full pipeline for Transformer:
        normalize batch -> reshape for Transformer

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = preprocess_batch(X)
    X = reshape_for_transformer(X)
    return X


def prepare_for_cnn_transformer(X):
    """
    Full pipeline for CNN + Transformer:
        normalize batch -> reshape for hybrid model

    Args:
        X (np.ndarray): shape (batch, length)

    Returns:
        np.ndarray: shape (batch, length, 1)
    """
    X = preprocess_batch(X)
    X = reshape_for_cnn_transformer(X)
    return X
