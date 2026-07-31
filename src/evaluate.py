import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import DEVICE, MODEL_PATH
from model import create_model
from dataset import get_dataloaders
from utils import load_model, save_model


def evaluate():

    _, val_loader = get_dataloaders()

    model = create_model().to(DEVICE)
    model = load_model(
        model,
        MODEL_PATH,
        DEVICE
    )

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    print("=" * 50)

    print(f"Accuracy : {accuracy_score(all_labels, all_predictions):.4f}")
    print(f"Precision: {precision_score(all_labels, all_predictions):.4f}")
    print(f"Recall   : {recall_score(all_labels, all_predictions):.4f}")
    print(f"F1 Score : {f1_score(all_labels, all_predictions):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(all_labels, all_predictions))

    print("\nClassification Report")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=["Mask", "No Mask"]
    ))


if __name__ == "__main__":
    evaluate()