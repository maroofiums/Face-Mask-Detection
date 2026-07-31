import torch
import matplotlib.pyplot as plt


def save_model(model, model_path):
    """
    Save model weights.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def load_model(model, model_path, device):
    """
    Load model weights.
    """
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")

    return model


def plot_history(train_loss, val_loss,
                 train_acc, val_acc):

    plt.figure(figsize=(8, 5))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")

    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.show()