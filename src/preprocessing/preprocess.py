import numpy as np



def normalize(signal):
    """
    Normalize ECG signal to zero mean and unit variance.
    """
    signal = np.asarray(signal)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)



def preprocess_batch(X):
    """
    Apply normalization to each signal in the batch.
    Args:
        X (np.ndarray): 2D array of shape (batch, length)
    Returns:
        np.ndarray: Normalized batch of same shape
    """
    X = np.asarray(X)
    return np.array([normalize(sig) for sig in X])



def reshape_for_cnn(X):
    """
    Reshape data for CNN: (batch, channel=1, length)
    Args:
        X (np.ndarray): 2D array (batch, length)
    Returns:
        np.ndarray: 3D array (batch, 1, length)
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], 1, X.shape[1])



def reshape_for_lstm(X):
    """
    Reshape for LSTM: (batch, sequence, features=1)
    Args:
        X (np.ndarray): 2D array (batch, length)
    Returns:
        np.ndarray: 3D array (batch, length, 1)
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], X.shape[1], 1)



def reshape_for_transformer(X):
    """
    Reshape for Transformer: (batch, sequence, features=1)
    Args:
        X (np.ndarray): 2D array (batch, length)
    Returns:
        np.ndarray: 3D array (batch, length, 1)
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], X.shape[1], 1)

print("preprocess completed")

def reshape_for_cnn_transformer(X):
    """
    Reshape for CNN + Transformer hybrid:
    Input shape: (batch, length)
    Output shape: (batch, length, 1)
    """
    X = np.asarray(X)
    return X.reshape(X.shape[0], X.shape[1], 1)