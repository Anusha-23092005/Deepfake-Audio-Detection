from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for AudioDeepfakeCNN.

    Extracts gradient-weighted activation maps from the final convolutional layer
    (model.features[8]: Conv2d(32, 64, kernel_size=3, padding=1)) to highlight
    the time-frequency regions influencing the model's classification decision.
    """
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        # Default to features[8] (last Conv2d layer before pooling)
        if target_layer is None:
            self.target_layer = self.model.features[8]
        else:
            self.target_layer = target_layer

        self.activations = []
        self.gradients = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates the 2D Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: Tensor of shape (1, 1, 80, 250)
            target_class: 0 for bonafide, 1 for spoof. If None, uses argmax class.

        Returns:
            cam_map: 2D numpy array of shape (80, 250) normalized to [0, 1]
            predicted_class: Integer class index
            confidence_score: Softmax probability of target class
        """
        self.activations.clear()
        self.gradients.clear()

        # Ensure gradients are tracked
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        confidence_score = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward()

        if not self.activations or not self.gradients:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        act = self.activations[0] # (1, 64, H_feat, W_feat)
        grad = self.gradients[0]  # (1, 64, H_feat, W_feat)

        # Channel-wise weights via global average pooling of gradients
        weights = torch.mean(grad, dim=(2, 3), keepdim=True) # (1, 64, 1, 1)
        cam = torch.sum(weights * act, dim=1, keepdim=True)   # (1, 1, H_feat, W_feat)
        cam = F.relu(cam) # Retain only positive contributions

        # Upsample heatmap to match input spectrogram dimensions (80, 250)
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_upsampled = F.interpolate(
            cam,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )

        cam_map = cam_upsampled.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_map), np.max(cam_map)
        if cam_max - cam_min > 1e-8:
            cam_map = (cam_map - cam_min) / (cam_max - cam_min)
        else:
            cam_map = np.zeros_like(cam_map)

        return cam_map, target_class, confidence_score

    def __del__(self):
        self.remove_hooks()


def plot_explainability(
    mel_db: np.ndarray,
    cam_map: np.ndarray,
    predicted_class_name: str,
    confidence_score: float,
    sample_rate: int = 16000,
    hop_length: int = 256
) -> plt.Figure:
    """
    Renders a dual-panel explainability plot:
    Panel 1: Original Log-Mel Spectrogram
    Panel 2: Grad-CAM Saliency Overlay

    Includes required scientific disclaimers distinguishing explanation from forensic proof.
    """
    time_extent_sec = (mel_db.shape[1] * hop_length) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # 1. Log-Mel Spectrogram
    im0 = axes[0].imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, time_extent_sec, 0, 80]
    )
    axes[0].set_title("Input Log-Mel Spectrogram (dB)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mel Frequency Bins", fontsize=10)
    cbar0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
    cbar0.set_label("dB", fontsize=9)

    # 2. Grad-CAM Overlay
    axes[1].imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="gray",
        alpha=0.4,
        extent=[0, time_extent_sec, 0, 80]
    )
    im1 = axes[1].imshow(
        cam_map,
        aspect="auto",
        origin="lower",
        cmap="jet",
        alpha=0.65,
        extent=[0, time_extent_sec, 0, 80]
    )
    axes[1].set_title(
        f"Grad-CAM Heatmap Overlay — Decision Influence for: {predicted_class_name.upper()} "
        f"(Model Softmax Score: {confidence_score * 100:.2f}%)",
        fontsize=11,
        fontweight="bold"
    )
    axes[1].set_xlabel("Time (seconds)", fontsize=10)
    axes[1].set_ylabel("Mel Frequency Bins", fontsize=10)
    cbar1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
    cbar1.set_label("Relative Saliency", fontsize=9)

    # Disclaimer footer
    fig.text(
        0.5, -0.04,
        "Note: Grad-CAM visualizes time-frequency features influencing this model's decision. "
        "It provides model explainability, NOT definitive forensic proof of manipulation.",
        ha="center",
        fontsize=8.5,
        style="italic",
        color="#555555"
    )

    return fig
