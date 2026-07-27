import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from config import NUM_CLASSES


def create_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model
