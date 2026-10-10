"""Run one local TIFF through inference.py and save a binary PNG mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio

from inference import DEFAULT_MODEL_PATH, WaterSegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", help="Path to one 12-band 128x128 TIFF/GeoTIFF")
    parser.add_argument("--output", default="outputs/predicted_mask.png", help="Output PNG path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to best_Unet.pt")
    args = parser.parse_args()

    input_path = Path(args.input_tiff).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    with rasterio.open(input_path) as src:
        image = src.read().astype(np.float32)
    segmenter = WaterSegmenter(model_path=args.model)
    mask = segmenter.predict_array(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(output_path)
    water_percent = 100 * float(mask.sum()) / mask.size
    print(f"Input shape: {image.shape}")
    print(f"Output saved: {output_path}")
    print(f"Mask values: {np.unique(mask).tolist()}")
    print(f"Predicted water coverage: {water_percent:.2f}%")


if __name__ == "__main__":
    main()
