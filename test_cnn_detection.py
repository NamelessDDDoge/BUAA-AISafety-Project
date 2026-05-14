"""Test script for CNNDetection (Wang et al. 2020) baseline models.

Supports blur_jpg_prob0.1.pth and blur_jpg_prob0.5.pth checkpoints.

Dataset source (mutually exclusive):
  --config   path to datasets.yaml (same format used by test.py); evaluates
             every entry in the YAML, including diffusion datasets that have
             separate real_path / fake_path.
  --dataroot scan every immediate subdirectory for 0_real/ + 1_fake/ images
             at any nesting depth.  Categories missing either class are skipped.

Results (per-category AP and Acc, plus overall mAP / mAcc) are written
to --result_folder.

Examples
--------
# YAML-driven (recommended, covers diffusion datasets correctly)
python test_cnn_detection.py \\
    --config data/datasets.yaml \\
    --model_path weights/blur_jpg_prob0.5.pth \\
    --result_folder result_cnn

# Directory scan (GAN-only datasets that contain self-paired 0_real/1_fake)
python test_cnn_detection.py \\
    --dataroot datasets/test \\
    --model_path weights/blur_jpg_prob0.5.pth \\
    --result_folder result_cnn
"""
import argparse
import os
import shutil

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score
from torchvision import transforms

from data.test_data import load_test_dataset_specs
from models import get_model
from utils.reproducibility import set_seed

SEED = 0
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _list_images(root, must_contain):
    results = []
    for dirpath, _, filenames in os.walk(root):
        if must_contain not in dirpath:
            continue
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                results.append(os.path.join(dirpath, fname))
    return results


class _RealFakeDataset(torch.utils.data.Dataset):
    def __init__(self, real_root, fake_root, transform):
        reals = _list_images(real_root, "0_real")
        fakes = _list_images(fake_root, "1_fake")
        if not reals:
            raise ValueError(f"No 0_real images found under {real_root}")
        if not fakes:
            raise ValueError(f"No 1_fake images found under {fake_root}")
        self.samples = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _get_dataloader_from_paths(real_root, fake_root, batch_size, num_workers):
    dataset = _RealFakeDataset(real_root, fake_root, transform=_get_transform())
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _load_weights(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    # Unwrap nested checkpoint format {'model': state_dict, 'optimizer': ..., ...}
    if isinstance(ckpt, dict) and "model" in ckpt and not any(
        isinstance(v, torch.Tensor) for v in ckpt.values()
    ):
        ckpt = ckpt["model"]
    cleaned = {}
    for k, v in ckpt.items():
        for prefix in ("model.", "module.", "net."):
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=False)
    return model


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(SEED)

    parser = argparse.ArgumentParser(
        description="CNNDetection Baseline Test (Wang et al. 2020)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataroot", default=None,
                        help="test dataset root; immediate subdirs are categories "
                             "(skips any subdir missing 0_real or 1_fake)")
    parser.add_argument("--config", default=None,
                        help="datasets.yaml path (preferred; covers diffusion datasets)")
    parser.add_argument("--model_path", required=True,
                        help="path to .pth checkpoint")
    parser.add_argument("--arch", type=str, default="CNNDetection:resnet50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (must be 0 on Windows)")
    parser.add_argument("--result_folder", type=str, default="result_cnn")
    opt = parser.parse_args()

    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    if opt.config is None and opt.dataroot is None:
        raise SystemExit("Provide either --config or --dataroot.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = get_model(opt)
    model = _load_weights(model, opt.model_path, device)
    model.to(device)
    model.eval()
    print(f"Model loaded: {opt.model_path}\n", flush=True)

    # Build (name, real_root, fake_root) list from yaml or directory scan
    if opt.config is not None:
        specs = load_test_dataset_specs(opt.config)
        eval_specs = [(s.key, s.real_path, s.fake_path) for s in specs]
        print(f"Loaded {len(eval_specs)} entries from {opt.config}\n", flush=True)
    else:
        subdirs = sorted([
            os.path.join(opt.dataroot, d)
            for d in os.listdir(opt.dataroot)
            if os.path.isdir(os.path.join(opt.dataroot, d))
        ])
        if not subdirs:
            raise SystemExit(f"No subdirectories found under: {opt.dataroot}")
        eval_specs = [(os.path.basename(d), d, d) for d in subdirs]
        print(f"Found {len(eval_specs)} subdirectories under {opt.dataroot}\n", flush=True)

    all_aps, all_accs = [], []

    with torch.no_grad():
        for idx, (cls_name, real_root, fake_root) in enumerate(eval_specs, start=1):
            try:
                loader = _get_dataloader_from_paths(
                    real_root, fake_root, opt.batch_size, opt.num_workers
                )
            except ValueError as exc:
                print(f"[{idx}/{len(eval_specs)}] {cls_name}: SKIP — {exc}", flush=True)
                continue

            print(f"[{idx}/{len(eval_specs)}] {cls_name}: {len(loader.dataset)} samples",
                  flush=True)

            logits_list, labels_list = [], []
            for imgs, lbls in loader:
                out = model(imgs.to(device)).flatten()
                logits_list.extend(out.cpu().tolist())
                labels_list.extend(lbls.tolist())

            y_true = np.array(labels_list, dtype=np.int32)
            y_logits = np.array(logits_list, dtype=np.float32)
            y_prob = 1.0 / (1.0 + np.exp(-y_logits))

            ap = average_precision_score(y_true, y_logits) * 100
            acc = ((y_prob > 0.5).astype(int) == y_true).mean() * 100

            print(f"  {cls_name:<25} AP: {ap:.2f}%  Acc: {acc:.2f}%", flush=True)
            all_aps.append(ap)
            all_accs.append(acc)

            with open(os.path.join(opt.result_folder, "ap.txt"), "a") as f:
                f.write(f"{cls_name}: {ap:.2f}\n")
            with open(os.path.join(opt.result_folder, "acc.txt"), "a") as f:
                f.write(f"{cls_name}: {acc:.2f}\n")

    mean_ap = np.mean(all_aps)
    mean_acc = np.mean(all_accs)
    print(f"\nmAP:  {mean_ap:.2f}%", flush=True)
    print(f"mAcc: {mean_acc:.2f}%", flush=True)

    with open(os.path.join(opt.result_folder, "summary.txt"), "w") as f:
        f.write(f"mAP:  {mean_ap:.2f}\n")
        f.write(f"mAcc: {mean_acc:.2f}\n")
