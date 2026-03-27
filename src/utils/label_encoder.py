import numpy as np

# AAMI Mapping
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',

    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',

    'V': 'V', 'E': 'V',

    'F': 'F',

    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

# Final class to number mapping
CLASS_MAP = {
    'N': 0,
    'S': 1,
    'V': 2,
    'F': 3,
    'Q': 4
}


def encode_labels(labels):
    """
    Convert raw MIT-BIH labels → AAMI classes → numeric labels
    """
    encoded = []

    for label in labels:
        if label in AAMI_MAP:
            mapped = AAMI_MAP[label]
            encoded.append(CLASS_MAP[mapped])
        else:
            # ignore unknown labels
            continue

    return np.array(encoded)


def filter_data(X, y):
    """
    Remove samples with unknown labels
    """
    filtered_X = []
    filtered_y = []

    for i in range(len(y)):
        if y[i] in AAMI_MAP:
            filtered_X.append(X[i])
            filtered_y.append(y[i])

    return np.array(filtered_X), np.array(filtered_y)