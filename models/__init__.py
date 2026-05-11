from .clip_binary import CLIPModel
from .imagenet_model import ImagenetModel

VALID_NAMES = [
    'Imagenet:resnet50',
    'Imagenet:vit_b_16',

    'CLIP:RN50', 
    'CLIP:ViT-L/14', 
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:])  
    else:
        assert False 
