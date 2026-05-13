from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import random
import numpy as np

from io import BytesIO
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from scipy.ndimage.filters import gaussian_filter

try:
    import torchvision.transforms as transforms
except ModuleNotFoundError:
    class _Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image):
            for transform in self.transforms:
                image = transform(image)
            return image

    class _CenterCrop:
        def __init__(self, size):
            self.size = (size, size) if isinstance(size, int) else size

        def __call__(self, image):
            crop_h, crop_w = self.size
            width, height = image.size
            left = int(round((width - crop_w) / 2.0))
            top = int(round((height - crop_h) / 2.0))
            return image.crop((left, top, left + crop_w, top + crop_h))

    class _ToTensor:
        def __call__(self, image):
            import torch

            array = np.array(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).permute(2, 0, 1)

    class _Normalize:
        def __init__(self, mean, std):
            import torch

            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, tensor):
            return (tensor - self.mean) / self.std

    class transforms:
        Compose = _Compose
        CenterCrop = _CenterCrop
        ToTensor = _ToTensor
        Normalize = _Normalize


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_CONFIG_PATH = Path(__file__).with_name("datasets.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


# 单个测试集配置：key 对应数据集名，real_path/fake_path 对应真假图像目录。
@dataclass(frozen=True)
class TestDatasetSpec:
    key: str
    real_path: str
    fake_path: str
    data_mode: str = "wang2020"


def load_test_dataset_specs(config_path=DEFAULT_CONFIG_PATH):
    """Load the small datasets.yaml format used by this repository."""
    # 从 data/datasets.yaml 读取所有测试集配置，返回 TestDatasetSpec 列表。
    config_path = Path(config_path)
    items = []
    current = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "datasets:":
            continue
        if line.startswith("- "):
            if current:
                items.append(current)
            current = {}
            line = line[2:].strip()
            if not line:
                continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {config_path}: {raw_line}")
        key, value = line.split(":", 1)
        if current is None:
            raise ValueError(f"Expected dataset item before line: {raw_line}")
        current[key.strip()] = value.strip().strip("\"'")

    if current:
        items.append(current)

    return [TestDatasetSpec(**item) for item in items]


def list_images(path, must_contain=""):
    # 用 must_contain 过滤 real/fake 两个类别
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    images = []
    for root, _, files in os.walk(path):
        if must_contain and must_contain not in root:
            continue
        for filename in files:
            if Path(filename).suffix.lower() not in IMAGE_EXTS:
                continue
            item = os.path.join(root, filename)
            if must_contain and must_contain not in item:
                continue
            images.append(item)
    return sorted(images)


def read_real_fake_paths(spec, max_sample=None, seed=0):
    # 按数据集模式读取真假图片路径；wang2020 约定文件名含 0_real 和 1_fake。
    real_paths = spec.real_path if isinstance(spec.real_path, list) else [spec.real_path]
    fake_paths = spec.fake_path if isinstance(spec.fake_path, list) else [spec.fake_path]

    reals = []
    fakes = []
    for real_path, fake_path in zip(real_paths, fake_paths):
        if spec.data_mode == "wang2020":
            reals += list_images(real_path, must_contain="0_real")
            fakes += list_images(fake_path, must_contain="1_fake")
        elif spec.data_mode == "ours":
            reals += list_images(real_path)
            fakes += list_images(fake_path)
        else:
            raise ValueError(f"Unsupported data_mode: {spec.data_mode}")
    
    if not reals or not fakes:
        raise ValueError(
            f"{spec.key} requires at least one real and one fake image; "
            f"found real={len(reals)} fake={len(fakes)}"
        )

    if max_sample is not None:
        if (max_sample > len(reals)) or (max_sample > len(fakes)):
            raise ValueError(
                f"Images not enough for {spec.key}: requested {max_sample} per class, "
                f"found real={len(reals)} fake={len(fakes)}"
            )
        rng = random.Random(seed)
        rng.shuffle(reals)
        rng.shuffle(fakes)
        reals = reals[:max_sample]
        fakes = fakes[:max_sample]

    return reals, fakes


class TestDataset(Dataset):
    """Test-only real/fake image reader. Labels: real=0, fake=1."""

    def __init__(self, spec, max_sample=None, seed=0, 
                 arch="res50",
                 jpeg_quality=None,
                 gaussian_sigma=None):
        self.spec = spec
        reals, fakes = read_real_fake_paths(spec, max_sample=max_sample, seed=seed)
        self.samples = [(path, 0) for path in reals] + [(path, 1) for path in fakes]
        self.total_list = reals + fakes
        self.labels_dict = {}
        for path in reals:
            self.labels_dict[path] = 0
        for path in fakes:
            self.labels_dict[path] = 1
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        path = self.total_list[idx]
        label = self.labels_dict[path]
        image = Image.open(path).convert("RGB")
        if self.gaussian_sigma is not None:
            image = self._gaussian_blur(image, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            image = self._png2jpg(image, self.jpeg_quality)
        image = self.transform(image)
        
        return image, label
    
    def _gaussian_blur(self, img, sigma):
        # 数据增强: 高斯模糊
        img = np.array(img)
        gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
        gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
        gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)
        return Image.fromarray(img)
    
    def _png2jpg(self, img, quality):
        # 数据增强: jpeg
        out = BytesIO()
        img.save(out, format='jpeg', quality=quality) 
        img = Image.open(out)
        img = np.array(img)
        out.close()
        return Image.fromarray(img)

if __name__ == "__main__":
    # 单元测试：运行 python data/test_data.py 检查导入和路径。
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test TestRealFakeDataset import and image loading.")
    parser.add_argument("key", nargs="?", default="progan", help="dataset key in data/datasets.yaml")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to datasets.yaml")
    parser.add_argument("--max-sample", type=int, default=1000, help="sample count per class for the smoke test")
    parser.add_argument("--arch", type=str, default="res50")
    args = parser.parse_args()

    specs = load_test_dataset_specs(args.config)
    spec_by_key = {spec.key: spec for spec in specs}
    if args.key not in spec_by_key:
        available = ", ".join(spec_by_key)
        raise SystemExit(f"Unknown dataset key: {args.key}. Available keys: {available}")

    spec = spec_by_key[args.key]
    real_paths, fake_paths = read_real_fake_paths(spec, max_sample=args.max_sample)
    dataset = TestDataset(spec, max_sample=args.max_sample, arch=args.arch)
    image, label = dataset[0]

    print(f"OK: {spec.key}")
    print(f"real={len(real_paths)} fake={len(fake_paths)} total={len(dataset)}")
    print(f"first_image_shape={tuple(image.shape)} first_label={label}")
