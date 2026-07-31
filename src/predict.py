import torch
from PIL import Image

from config import DEVICE, MODEL_PATH, CLASS_NAMES
from model import create_model
from transforms import test_transforms
from utils import load_model


def predict(image_path):
    model = create_model().to(DEVICE)
    model = load_model(
        model,
        MODEL_PATH,
        DEVICE
    )

    image = Image.open(image_path).convert("RGB")
    image = test_transformsa(image)
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    print(f"Prediction: {CLASS_NAMES[prediction.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}")