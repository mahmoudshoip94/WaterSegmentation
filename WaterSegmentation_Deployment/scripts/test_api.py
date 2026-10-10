"""Smoke-test a running API on at least three real TIFFs.

Example (run from the deployment project root):
  python scripts/test_api.py --images-dir ../data/images --count 3
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import rasterio
import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]


def default_images_dir() -> Path:
    candidates = [
        BASE_DIR.parent / "data" / "images",  # deployment folder inside repo root
        BASE_DIR / "data" / "images",         # deployment files placed directly at repo root
        BASE_DIR.parent.parent / "data" / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default=str(default_images_dir()))
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output-dir", default=str(BASE_DIR / "outputs" / "api_tests"))
    args = parser.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    tiffs = sorted([*images_dir.glob("*.tif"), *images_dir.glob("*.tiff"),
                    *images_dir.glob("*.TIF"), *images_dir.glob("*.TIFF")])
    # Deduplicate paths in case a case-insensitive filesystem matches both patterns.
    tiffs = list(dict.fromkeys(tiffs))
    if len(tiffs) < args.count:
        raise SystemExit(f"Found {len(tiffs)} TIFFs in {images_dir}; need {args.count}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for path in tiffs[:args.count]:
        try:
            with rasterio.open(path) as src:
                expected_size = (src.width, src.height)
                band_count = src.count
            with path.open("rb") as file_handle:
                response = requests.post(
                    args.url,
                    files={"file": (path.name, file_handle, "image/tiff")},
                    timeout=180,
                )

            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            if response.headers.get("Content-Type", "").split(";")[0] != "image/png":
                raise RuntimeError(f"Expected image/png, got {response.headers.get('Content-Type')}")
            if band_count != 12:
                raise RuntimeError(f"Input has {band_count} bands, not 12.")

            mask = Image.open(io.BytesIO(response.content)).convert("L")
            if mask.size != expected_size:
                raise RuntimeError(f"Output dimensions {mask.size} != input dimensions {expected_size}.")
            values = set(mask.getdata())
            if not values.issubset({0, 255}):
                raise RuntimeError(f"Output PNG is not binary: found values such as {sorted(values)[:10]}.")

            out_path = output_dir / f"{path.stem}_water_mask.png"
            mask.save(out_path)
            print(
                f"PASS | {path.name} | bands={band_count} | size={mask.width}x{mask.height} "
                f"| output values={sorted(values)} | water={response.headers.get('X-Water-Pixel-Percent', 'n/a')}% "
                f"| saved={out_path}"
            )
        except Exception as exc:
            failed.append((path.name, str(exc)))
            print(f"FAIL | {path.name} | {exc}")

    if failed:
        raise SystemExit(f"{len(failed)} of {args.count} tests failed.")
    print(f"\nAll {args.count} API smoke tests passed.")


if __name__ == "__main__":
    main()
