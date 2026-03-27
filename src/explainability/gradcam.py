import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # ✅ FIXED hooks
        self.target_layer.register_full_backward_hook(self.save_gradients)
        self.target_layer.register_forward_hook(self.save_activations)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate_explanation(pred_class):
        explanations = {
            0: "The model predicts a NORMAL beat. It focuses on regular and consistent waveform patterns, indicating stable cardiac activity.",

            1: "The model predicts SUPRAVENTRICULAR arrhythmia. It detects irregular timing and slight waveform distortions.",

            2: "The model predicts VENTRICULAR arrhythmia. It focuses on abnormal spikes and distorted QRS complexes.",

            3: "The model predicts a FUSION beat. It detects mixed waveform characteristics from normal and abnormal signals.",

            4: "The model predicts UNKNOWN class. The signal does not clearly match known arrhythmia patterns."
        }

        return explanations.get(pred_class, "No explanation available.")

    def save_activations(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        target = output[:, class_idx]

        target.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=1)

        cam = torch.zeros(activations.shape[1])

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam