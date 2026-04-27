import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn

# Load data
X, y = load_dataset()

# Filter + encode labels
X, y = filter_data(X, y)
y_encoded = encode_labels(y)

# Preprocess signals
X_processed = preprocess_batch(X)

# Reshape for CNN
X_final = reshape_for_cnn(X_processed)

print("Final shape:", X_final.shape)
print("Labels shape:", y_encoded.shape)
