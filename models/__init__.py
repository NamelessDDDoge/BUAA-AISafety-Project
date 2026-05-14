VALID_NAMES = [
    'Imagenet:resnet50',
    'Imagenet:vit_b_16',

    'CLIP:RN50',
    'CLIP:ViT-L/14',

    'CNNDetection:resnet50',
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        from .imagenet_model import ImagenetModel

        return ImagenetModel(name[9:])
    elif name.startswith("CLIP:"):
        from .clip_binary import CLIPModel

        return CLIPModel(name[5:])
    elif name.startswith("CNNDetection:"):
        from .cnn_detection import get_cnn_detection_model

        return get_cnn_detection_model()
    else:
        assert False
