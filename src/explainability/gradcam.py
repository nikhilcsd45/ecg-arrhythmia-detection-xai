from __future__ import annotations

import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_full_backward_hook(self.save_gradients)
        self.target_layer.register_forward_hook(self.save_activations)

    def save_gradients(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activations(self, _module, _inputs, output):
        self.activations = output

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        target = output[:, class_idx]

        target.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=1)
        cam = torch.zeros(activations.shape[1], device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
