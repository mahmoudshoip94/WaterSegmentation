"""Flask app for water segmentation from 12-band TIFF files."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np
import rasterio
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from rasterio.io import MemoryFile

from inference import DEFAULT_MODEL_PATH, WaterSegmenter, make_rgb_preview

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".tif", ".tiff"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("water-segmentation")

_segmenter = None
_segmenter_lock = Lock()


def get_segmenter() -> WaterSegmenter:
    """Load the model only on the first prediction request."""
    global _segmenter
    if _segmenter is None:
        with _segmenter_lock:
            if _segmenter is None:
                model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
                logger.info("Loading segmentation model from %s", model_path)
                _segmenter = WaterSegmenter(model_path=model_path)
                logger.info("Model loaded on %s", _segmenter.device)
    return _segmenter


def get_uploaded_tiff():
    """Return (uploaded_file, bytes) or a Flask error response tuple."""
    if "file" not in request.files:
        return None, None, (jsonify(error="Missing upload field 'file'."), 400)

    uploaded = request.files["file"]
    if not uploaded.filename:
        return None, None, (jsonify(error="No file was selected."), 400)

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return None, None, (jsonify(error="Unsupported extension. Please upload a .tif or .tiff file."), 400)

    file_bytes = uploaded.read()
    if not file_bytes:
        return None, None, (jsonify(error="The uploaded file is empty."), 400)
    return uploaded, file_bytes, None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()
    return jsonify(
        status="ok" if model_path.is_file() else "model_missing",
        model_file_exists=model_path.is_file(),
        model_loaded=_segmenter is not None,
        device=str(_segmenter.device) if _segmenter is not None else None,
    ), (200 if model_path.is_file() else 503)


@app.post("/preview")
def preview():
    """Return a display-only RGB PNG made from bands 4, 3, 2 (indices 3,2,1)."""
    uploaded, file_bytes, error = get_uploaded_tiff()
    if error:
        return error
    try:
        with MemoryFile(file_bytes) as memory_file:
            with memory_file.open() as dataset:
                raw = dataset.read().astype(np.float32)
        rgb = make_rgb_preview(raw)
        output = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(output, format="PNG")
        output.seek(0)
        return send_file(output, mimetype="image/png", download_name="rgb_preview.png")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        logger.exception("Failed to generate RGB preview for %s", uploaded.filename)
        return jsonify(error="Could not generate preview from this TIFF."), 400


@app.post("/predict")
def predict():
    """Accept multipart/form-data field 'file' and return a binary PNG mask."""
    uploaded, file_bytes, error = get_uploaded_tiff()
    if error:
        return error

    try:
        segmenter = get_segmenter()
        mask = segmenter.predict_bytes(file_bytes)  # uint8 values {0,1}
        water_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        water_percent = 100.0 * water_pixels / max(total_pixels, 1)

        output = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(output, format="PNG")
        output.seek(0)
        response = send_file(
            output,
            mimetype="image/png",
            as_attachment=True,
            download_name="water_mask.png",
            max_age=0,
        )
        response.headers["X-Water-Pixel-Percent"] = f"{water_percent:.2f}"
        response.headers["X-Water-Pixel-Count"] = str(water_pixels)
        response.headers["X-Total-Pixel-Count"] = str(total_pixels)
        response.headers["X-Model-Threshold"] = "0.5"
        return response
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except FileNotFoundError as exc:
        logger.exception("Model checkpoint missing")
        return jsonify(error=str(exc)), 503
    except Exception:
        logger.exception("Prediction failed for %s", uploaded.filename)
        return jsonify(error="Prediction failed. Check the server console and model checkpoint."), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error=f"Upload is too large. Maximum size is {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    # For local development only. For production, run via a production WSGI server.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
