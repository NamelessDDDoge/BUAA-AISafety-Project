import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck


class CNNDetectionResNet50(ResNet):
    """ResNet-50 for CNNDetection (Wang et al. 2020).

    Inherits directly from torchvision ResNet so checkpoint key names align
    with the original blur_jpg_prob*.pth weights without any prefix stripping.
    """
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])
        self.fc = nn.Linear(2048, 1)


def get_cnn_detection_model():
    return CNNDetectionResNet50()
