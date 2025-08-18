from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


class ReIDModel(nn.Module):
    """
    ReID feature extractor built on ResNet-50.

    - Expects a single BGR image (NumPy array).
    - Outputs a L2-normalized 1D embedding (float32).
    """

    device: torch.device
    feature_extractor: nn.Sequential
    transform: T.Compose

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """
        Args:
            device: Optional target device; if None, picks CUDA when available.
        """
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load ImageNet-pretrained ResNet-50 and drop the classifier head.
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze params (we use it only for inference).
        for p in base_model.parameters():
            p.requires_grad = False

        # Keep everything up to the global pooling (avgpool) layer.
        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()  # inference mode

        # Standard person re-id input size (H, W) ~ (256, 128) and ImageNet normalization.
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def get_embedding(self, image_bgr: NDArray[Any]) -> FloatArray:
        """
        Convert a BGR image (H x W x 3, uint8) to a normalized feature vector.

        Args:
            image_bgr: OpenCV-style BGR image as a NumPy array.

        Returns:
            1D L2-normalized embedding as np.float32 (shape: [2048]).
        """
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be an HxWx3 NumPy array.")

        # OpenCV -> RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess and move to device.
        inp = self.transform(image_rgb).unsqueeze(
            0).to(self.device)  # [1, 3, 256, 128]

        # Forward to get [1, 2048, 1, 1] then flatten to [2048]
        feat = self.feature_extractor(inp)
        feat = torch.flatten(feat, 1)  # [1, 2048]

        # Convert to NumPy and normalize to unit length.
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm

        return vec
