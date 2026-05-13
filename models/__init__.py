VALID_NAMES = [
    "Imagenet:resnet50",
    "Imagenet:vit_b_16",
    "CLIP:RN50",
    "CLIP:ViT-L/14",
    "xception",
]


def get_model(opts):
    name = opts.arch
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        from .imagenet_model import ImagenetModel

        return ImagenetModel(name[9:])
    elif name.startswith("CLIP:"):
        from .clip_binary import CLIPModel

        return CLIPModel(name[5:])
    elif name == "xception":
        from .patch_classify_model import PatchDiscriminatorModel

        return PatchDiscriminatorModel(opts)
    else:
        assert False
