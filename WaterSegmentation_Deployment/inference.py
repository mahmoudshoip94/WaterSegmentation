"""Inference utilities for the 12-channel Water Segmentation model.

Expected input: 12-band GeoTIFF/TIFF, 128 x 128 pixels, bands in the
project's documented order. Output mask: uint8 array with 0=background,
1=water. The API converts this to an 8-bit PNG (0=black, 255=white).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from rasterio.io import MemoryFile

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "best_Unet.pt"

NUM_CHANNELS = 12
EXPECTED_HEIGHT = 128
EXPECTED_WIDTH = 128
THRESHOLD = 0.5

# Only these continuous channels are standardized. QA (7) and
# ESA WorldCover (10) are intentionally left unchanged.
NORM_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]
TRAIN_MEAN = np.asarray(
    [394.15018, 492.13864, 820.9418, 973.08264, 2069.7773,
     1948.11, 1336.509, 123.91796, 294.36536, 10.063989],
    dtype=np.float32,
)
TRAIN_STD = np.asarray(
    [261.87573, 310.67526, 396.09274, 567.03094, 1070.1465,
     1196.9388, 953.8549, 1394.3047, 456.21017, 28.180136],
    dtype=np.float32,
)


class WaterSegmenter:
    """Build the same model topology and load a trained state_dict."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.model_path}. "
                "Copy best_Unet.pt into the deployment model/ directory "
                "or set MODEL_PATH to its actual path."
            )

        requested_device = device or os.getenv("DEVICE")
        if requested_device:
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder_weights=None is intentional: all trained encoder weights,
        # including the adapted 12-channel conv1, are loaded from the checkpoint.
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=NUM_CHANNELS,
            classes=1,
            activation=None,
        )

        # The project checkpoint was verified as a plain OrderedDict state_dict.
        # weights_only=True avoids loading arbitrary Python objects.
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "This deployment expects a PyTorch version that supports "
                "torch.load(..., weights_only=True). Upgrade PyTorch to >=2.4."
            ) from exc

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError("Unsupported checkpoint format: expected a state_dict.")

        # Support checkpoints saved from DataParallel without changing normal keys.
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {str(key)[7:]: value for key, value in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint keys/shapes do not match U-Net + ResNet34 with "
                "12 input channels. Confirm that best_Unet.pt belongs to the "
                "12-channel pretrained baseline, not the CNN Converter or 13-channel experiment."
            ) from exc

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def validate_array(image: np.ndarray) -> np.ndarray:
        """Validate and convert a raw raster array to float32 (C,H,W)."""
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(
                f"Expected a 3-D array in (channels, height, width) order; got shape {image.shape}."
            )
        if image.shape[0] != NUM_CHANNELS:
            raise ValueError(
                f"Expected exactly {NUM_CHANNELS} TIFF bands, but got {image.shape[0]}. "
                "An ordinary 3-channel RGB image cannot be used."
            )
        if image.shape[1:] != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
            raise ValueError(
                f"Expected spatial dimensions {EXPECTED_HEIGHT}x{EXPECTED_WIDTH}; "
                f"got {image.shape[1]}x{image.shape[2]}."
            )

        image = image.astype(np.float32, copy=False)
        if not np.isfinite(image).all():
            raise ValueError("Input TIFF contains NaN or infinite values. Please check the raster data.")
        return np.ascontiguousarray(image)

    @staticmethod
    def preprocess(image: np.ndarray) -> torch.Tensor:
        """Apply the same channel-wise standardization used by the notebook."""
        image = WaterSegmenter.validate_array(image)
        image_tensor = torch.from_numpy(image.copy()).to(dtype=torch.float32)

        norm_indices = torch.tensor(NORM_CHANNELS, dtype=torch.long)
        mean = torch.from_numpy(TRAIN_MEAN).view(-1, 1, 1)
        std = torch.from_numpy(TRAIN_STD).view(-1, 1, 1)

        # Keep channel 7 (QA) and 10 (ESA WorldCover) raw, matching training.
        image_tensor[norm_indices] = (image_tensor[norm_indices] - mean) / std
        return image_tensor.unsqueeze(0)  # (1, 12, 128, 128)

    @torch.inference_mode()
    def predict_array(self, image: np.ndarray) -> np.ndarray:
        """Predict a raw (12,128,128) array; return a 0/1 uint8 mask."""
        input_tensor = self.preprocess(image).to(self.device)
        logits = self.model(input_tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities > THRESHOLD).to(torch.uint8)[0, 0]
        return mask.cpu().numpy()

    @staticmethod
    def read_tiff_bytes(file_bytes: bytes) -> np.ndarray:
        """Read TIFF bytes without writing an uploaded file to disk."""
        if not file_bytes:
            raise ValueError("The uploaded file is empty.")
        try:
            with MemoryFile(file_bytes) as memory_file:
                with memory_file.open() as dataset:
                    image = dataset.read()
        except Exception as exc:
            raise ValueError("Could not read this file as a supported TIFF/GeoTIFF raster.") from exc
        return WaterSegmenter.validate_array(image)

    def predict_bytes(self, file_bytes: bytes) -> np.ndarray:
        """Read uploaded TIFF bytes and return a 0/1 binary mask."""
        image = self.read_tiff_bytes(file_bytes)
        return self.predict_array(image)


def make_rgb_preview(image: np.ndarray) -> np.ndarray:
    """Create a display-only RGB preview from raw channels [Red, Green, Blue].

    This per-image percentile stretch is for visualization only; it is never
    fed into the model and does not alter the 12-channel inference input.
    Returns uint8 RGB (H,W,3).
    """
    image = WaterSegmenter.validate_array(image)
    rgb = np.stack([image[3], image[2], image[1]], axis=-1).astype(np.float32)
    channels = []
    for channel_index in range(3):
        channel = rgb[..., channel_index]
        low, high = np.percentile(channel, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            stretched = np.zeros_like(channel, dtype=np.uint8)
        else:
            stretched = np.clip((channel - low) / (high - low), 0, 1)
            stretched = (stretched * 255).round().astype(np.uint8)
        channels.append(stretched)
    return np.stack(channels, axis=-1)
