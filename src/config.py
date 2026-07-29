import os
import torch


DATASET_PATH = "dataset"


IMAGE_SIZE = 224
NUM_CLASSES = 2


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

CLASS_NAMES = [
    "with_mask",
    "without_mask"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

os.makedirs(MODEL_DIR, exist_ok=True)