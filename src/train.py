import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_PATH
from dataset import get_dataloaders
from model import create_model


def train():
    train_loader, val_loader = get_dataloaders()

    model = create_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader)

        for images, labels in progress_bar:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)

            train_correct += (predicted == labels).sum().item()

            progress_bar.set_description(
                f"Epoch {epoch+1}/{EPOCHS}"
            )

        train_accuracy = 100 * train_correct / train_total

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch [{epoch+1}/{EPOCHS}]"
            f"Train Accuracy: {train_accuracy:.2f}%"
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            save_model(model, MODEL_PATH)

            print("Best Model Saved!")

    print("Training Completed!...")
        

if __name__ == "__main__":
    train()