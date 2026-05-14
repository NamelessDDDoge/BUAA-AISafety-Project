"""Test script for CNNDetection (Wang et al. 2020) baseline models.

Supports blur_jpg_prob0.1.pth and blur_jpg_prob0.5.pth checkpoints.

--dataroot must point to a directory whose immediate subdirectories are
test categories (e.g. biggan/, progan/, ...).  Each category folder must
follow the ImageFolder layout expected by torchvision:

    <dataroot>/<category>/0_real/<images>
    <dataroot>/<category>/1_fake/<images>

Results (per-category AP and Acc, plus overall mAP / mAcc) are written
to --result_folder.

Example
-------
python test_cnn_detection.py \\
    --dataroot datasets/test \\
    --model_path /path/to/blur_jpg_prob0.5.pth \\
    --result_folder result_cnn
"""
import argparse
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torchvision import datasets, transforms

from models import get_model
from utils.reproducibility import set_seed

SEED = 0


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


def _get_dataloader(root, batch_size, num_workers):
    dataset = datasets.ImageFolder(root=root, transform=_get_transform())
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def _load_weights(model, model_path, device):
    """Load checkpoint, stripping common multi-GPU prefixes."""
    ckpt = torch.load(model_path, map_location=device)
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
    parser.add_argument("--dataroot", required=True,
                        help="test dataset root; immediate subdirs are categories")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = get_model(opt.arch)
    model = _load_weights(model, opt.model_path, device)
    model.to(device)
    model.eval()
    print(f"Model loaded: {opt.model_path}\n", flush=True)

    cls_dirs = sorted([
        os.path.join(opt.dataroot, d)
        for d in os.listdir(opt.dataroot)
        if os.path.isdir(os.path.join(opt.dataroot, d))
    ])
    if not cls_dirs:
        raise SystemExit(f"No category subdirectories found under: {opt.dataroot}")

    print(f"Found {len(cls_dirs)} categories: {[os.path.basename(d) for d in cls_dirs]}\n",
          flush=True)

    all_aps, all_accs = [], []

    with torch.no_grad():
        for idx, cls_path in enumerate(cls_dirs, start=1):
            cls_name = os.path.basename(cls_path)
            loader = _get_dataloader(cls_path, opt.batch_size, opt.num_workers)
            print(f"[{idx}/{len(cls_dirs)}] {cls_name}: {len(loader.dataset)} samples",
                  flush=True)

            logits_list, labels_list = [], []
            for imgs, lbls in loader:
                out = model(imgs.to(device)).flatten()
                logits_list.extend(out.cpu().numpy())
                labels_list.extend(lbls.numpy())

            y_true = np.array(labels_list)
            y_logits = np.array(logits_list)
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
