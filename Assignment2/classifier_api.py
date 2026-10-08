from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from helper_lib.data_loader import CLASSES, transform
from helper_lib.model import CNN

router = APIRouter()
MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "cifar10_cnn.pth"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Train the model first: python train.py")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = CNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@router.post("/classify")
def classify(file: UploadFile = File(...)):
    contents = file.file.read(10 * 1024 * 1024 + 1)
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload an image smaller than 10 MB.")
    try:
        with Image.open(BytesIO(contents)) as image:
            inputs = transform(image.convert("RGB")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="Upload a valid image.")

    model = load_model()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    index = probabilities.argmax().item()
    return {"class_id": index, "class_name": CLASSES[index],
            "probability": probabilities[index].item()}
