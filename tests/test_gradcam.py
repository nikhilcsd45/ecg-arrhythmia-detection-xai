import sys
import os
import numpy as np

# Add root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt

from src.models.cnn import CNNModel
from src.data.loader import load_dataset
from src.utils.label_encoder import encode_labels, filter_data
from src.preprocessing.preprocess import preprocess_batch, reshape_for_cnn
from src.explainability.gradcam import GradCAM


# 🔥 Explanation function
def generate_explanation(predicted_class):
    explanations = {
        0: (
            "The model predicts a NORMAL beat. It focuses on regular and "
            "consistent waveform patterns, indicating stable cardiac activity."
        ),
        1: (
            "The model predicts SUPRAVENTRICULAR arrhythmia. It detects "
            "irregular timing and slight waveform distortions."
        ),
        2: (
            "The model predicts VENTRICULAR arrhythmia. It focuses on "
            "abnormal spikes and distorted QRS complexes."
        ),
        3: (
            "The model predicts a FUSION beat. It detects mixed waveform "
            "characteristics from normal and abnormal signals."
        ),
        4: (
            "The model predicts UNKNOWN class. The signal does not clearly "
            "match known arrhythmia patterns."
        ),
    }

    return explanations.get(predicted_class, "No explanation available.")


# 🔹 STEP 1: Load Data
print("Loading data...")
X, y = load_dataset()
X, y = filter_data(X, y)
y = encode_labels(y)

X = preprocess_batch(X)
X = reshape_for_cnn(X)

# 🔹 STEP 2: Take one sample
sample = torch.tensor(X[0:1]).float()
true_label = y[0]

# 🔹 STEP 3: Load Model
model = CNNModel()
model.load_state_dict(torch.load("models/cnn_weighted.pth"))
model.eval()

# 🔹 STEP 4: Setup Grad-CAM
target_layer = model.conv2
gradcam = GradCAM(model, target_layer)

# 🔹 STEP 5: Predict
output = model(sample)
pred_class = torch.argmax(output, dim=1).item()

print(f"\nTrue Label: {true_label}, Predicted: {pred_class}")

# 🔥 PRINT EXPLANATION
explanation = generate_explanation(pred_class)

print("\n🧠 Explanation:")
print(explanation)

# 🔹 STEP 6: Generate Heatmap
heatmap = gradcam.generate(sample, pred_class)

# 🔹 STEP 7: Get signal
signal = sample.numpy()[0][0]

# 🔥 Resize heatmap to match signal length
heatmap = np.interp(
    np.linspace(0, len(heatmap) - 1, num=len(signal)),
    np.arange(len(heatmap)),
    heatmap
)

# 🔹 STEP 8: Plot
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(12, 4))

plt.plot(signal, color='blue', label="ECG Signal")

# 🔥 Highlight important region
plt.fill_between(
    range(len(signal)),
    heatmap * max(signal),
    color='red',
    alpha=0.4,
    label="Important Region"
)

plt.title(f"True: {true_label}, Predicted: {pred_class}")
plt.legend()

# ✅ Save image
save_path = "outputs/gradcam_sample.png"
plt.savefig(save_path)

print(f"\n📊 Grad-CAM image saved at: {save_path}")
