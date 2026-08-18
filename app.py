import io
import os
import threading
from pathlib import Path


import requests
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torchvision.models import convnext_tiny




# ============================================================
# Configuration
# ============================================================


MODEL_PATH = Path(
    os.environ.get("GEOGUESSR_MODEL_PATH", "best_convnext_tiny_V3.pt")
)


STREET_VIEW_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
)


# This exactly matches your training-image preprocessing.
CROP_BOX = (0, 0, 512, 256)
SCRAPER_OUTPUT_SIZE = (320, 160)  # PIL uses (width, height)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




# ============================================================
# Flask setup
# ============================================================


app = Flask(__name__)


# Allows the Chrome extension to call the Flask API.
CORS(app)




# ============================================================
# Load the model once
# ============================================================


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find model checkpoint: {MODEL_PATH.resolve()}"
    )


print(f"Loading model from: {MODEL_PATH.resolve()}")
print(f"Using device: {DEVICE}")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)


class_names = checkpoint["class_names"]
image_height = checkpoint["image_height"]
image_width = checkpoint["image_width"]


model = convnext_tiny(
    weights=None,
    num_classes=len(class_names),
)


model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


print(f"Loaded classes: {class_names}")
print(f"Model input size: {image_width}x{image_height}")




# This matches the transform used in your single-image script.
model_transform = transforms.Compose(
    [
        transforms.Resize(
            (image_height, image_width),
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)




# Reuse HTTP connections instead of opening a completely new
# connection for every prediction request.
http_session = requests.Session()


# Prevent simultaneous requests from attempting GPU inference
# at exactly the same time.
inference_lock = threading.Lock()




# ============================================================
# Helper functions
# ============================================================


def extract_pano_id() -> str | None:
    """
    Accept pano IDs in several formats.


    JSON examples:
        {"pano_id": "abc123"}
        {"panoId": "abc123"}
        {"panoid": "abc123"}


    Query parameter example:
        /predict?pano_id=abc123
    """


    payload = request.get_json(silent=True)


    pano_id = None


    if isinstance(payload, dict):
        pano_id = (
            payload.get("pano_id")
            or payload.get("panoId")
            or payload.get("panoid")
        )


    if not pano_id:
        pano_id = (
            request.args.get("pano_id")
            or request.args.get("panoId")
            or request.args.get("panoid")
        )


    if not pano_id:
        pano_id = (
            request.form.get("pano_id")
            or request.form.get("panoId")
            or request.form.get("panoid")
        )


    if not pano_id:
        return None


    return str(pano_id).strip()




def fetch_panorama_image(pano_id: str) -> Image.Image:
    """
    Download the panorama tile and apply the exact same crop
    and resize operations used by the training-image scraper.
    """


    parameters = {
        "cb_client": "apiv3",
        "panoid": pano_id,
        "output": "tile",
        "x": 0,
        "y": 0,
        "zoom": 0,
        "nbt": 1,
        "fover": 2,
    }


    response = http_session.get(
        STREET_VIEW_TILE_URL,
        params=parameters,
        timeout=(5, 15),
    )


    response.raise_for_status()


    content_type = response.headers.get("Content-Type", "")


    if "image" not in content_type.lower():
        response_preview = response.text[:200]


        raise ValueError(
            "Street View returned something other than an image. "
            f"Content-Type: {content_type}. "
            f"Response: {response_preview}"
        )


    try:
        with Image.open(io.BytesIO(response.content)) as downloaded_image:
            image = downloaded_image.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError(
            "The Street View response could not be decoded as an image."
        ) from error


    if image.width < CROP_BOX[2] or image.height < CROP_BOX[3]:
        raise ValueError(
            "The downloaded Street View image was smaller than expected. "
            f"Received {image.width}x{image.height}, but the crop requires "
            f"at least {CROP_BOX[2]}x{CROP_BOX[3]}."
        )


    # Same crop used when creating the dataset.
    image = image.crop(CROP_BOX)


    # Same resize used when creating the dataset.
    image = image.resize(
        SCRAPER_OUTPUT_SIZE,
        Image.Resampling.BILINEAR,
    )


    return image




def predict_country(image: Image.Image) -> dict:
    """
    Run inference and return the predicted country, confidence,
    and probability for every class.
    """


    image_tensor = model_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)


    with inference_lock:
        with torch.inference_mode():
            logits = model(image_tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]


    probabilities = probabilities_tensor.detach().cpu().tolist()


    ranked_predictions = sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )


    predicted_country, confidence = ranked_predictions[0]


    return {
        "predicted_country": predicted_country,
        "confidence": confidence,
        "probabilities": {
            country: probability
            for country, probability in ranked_predictions
        },
    }




def print_prediction(pano_id: str, prediction: dict) -> None:
    print("\n" + "=" * 50)
    print(f"Panorama ID: {pano_id}")
    print(
        f"Prediction: {prediction['predicted_country']} "
        f"({prediction['confidence']:.2%})"
    )
    print("-" * 50)


    for country, probability in prediction["probabilities"].items():
        print(f"{country}: {probability:.2%}")


    print("=" * 50, flush=True)




# ============================================================
# API routes
# ============================================================


@app.get("/")
def index():
    return jsonify(
        {
            "status": "online",
            "device": str(DEVICE),
            "classes": class_names,
            "endpoint": "POST /predict",
            "example_body": {
                "pano_id": "your_panorama_id"
            },
        }
    )




@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "device": str(DEVICE),
        }
    )




@app.route("/predict", methods=["POST", "GET"])
def predict():
    pano_id = extract_pano_id()


    if not pano_id:
        return (
            jsonify(
                {
                    "success": False,
                    "error": (
                        "No panorama ID was provided. Send a JSON body "
                        'such as {"pano_id": "PANORAMA_ID"}.'
                    ),
                }
            ),
            400,
        )


    try:
        image = fetch_panorama_image(pano_id)
        prediction = predict_country(image)


        print_prediction(pano_id, prediction)


        return jsonify(
            {
                "success": True,
                "pano_id": pano_id,
                **prediction,
            }
        )


    except requests.Timeout:
        return (
            jsonify(
                {
                    "success": False,
                    "pano_id": pano_id,
                    "error": "The Street View image request timed out.",
                }
            ),
            504,
        )


    except requests.HTTPError as error:
        status_code = error.response.status_code


        return (
            jsonify(
                {
                    "success": False,
                    "pano_id": pano_id,
                    "error": (
                        "Street View returned an HTTP error: "
                        f"{status_code}"
                    ),
                }
            ),
            502,
        )


    except requests.RequestException as error:
        return (
            jsonify(
                {
                    "success": False,
                    "pano_id": pano_id,
                    "error": f"Street View request failed: {error}",
                }
            ),
            502,
        )


    except ValueError as error:
        return (
            jsonify(
                {
                    "success": False,
                    "pano_id": pano_id,
                    "error": str(error),
                }
            ),
            422,
        )


    except Exception:
        app.logger.exception(
            "Unexpected error while processing panorama %s",
            pano_id,
        )


        return (
            jsonify(
                {
                    "success": False,
                    "pano_id": pano_id,
                    "error": "An unexpected server error occurred.",
                }
            ),
            500,
        )


# start server


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,


        # prevent Flask's reloader from loading the model twice
        use_reloader=False,
    )

