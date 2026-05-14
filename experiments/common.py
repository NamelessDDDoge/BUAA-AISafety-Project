"""
Shared utilities for all experiments.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path before any project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Dataset taxonomy
# ---------------------------------------------------------------------------
DATASET_GROUPS = {
    # GAN-based
    "progan":    "GAN",
    "cyclegan":  "GAN",
    "biggan":    "GAN",
    "stylegan":  "GAN",
    "gaugan":    "GAN",
    "stargan":   "GAN",
    "deepfake":  "GAN",
    "sitd":      "GAN",
    "san":       "GAN",
    "crn":       "GAN",
    "imle":      "GAN",
    # Diffusion-based
    "ldm_200":      "Diffusion",
    "ldm_200_cfg":  "Diffusion",
    "ldm_100":      "Diffusion",
    "glide_100_27": "Diffusion",
    "glide_50_27":  "Diffusion",
    "glide_100_10": "Diffusion",
    "guided":       "Diffusion",
    # Autoregressive
    "dalle": "Autoregressive",
}

GROUP_COLORS = {
    "GAN":           "#2196F3",
    "Diffusion":     "#4CAF50",
    "Autoregressive": "#FF9800",
}

GROUP_ORDER = ["GAN", "Diffusion", "Autoregressive"]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Spec loading helpers
# ---------------------------------------------------------------------------

def get_available_specs(config_path=None):
    """
    Load datasets.yaml, check which paths actually exist on disk,
    and return the list of available TestDatasetSpec objects.
    Resolves relative paths against PROJECT_ROOT.
    """
    from data.test_data import load_test_dataset_specs, TestDatasetSpec

    if config_path is None:
        config_path = PROJECT_ROOT / "data" / "datasets.yaml"

    all_specs = load_test_dataset_specs(config_path)
    available = []

    for spec in all_specs:
        # Resolve real_path and fake_path relative to PROJECT_ROOT if needed
        real_p = Path(spec.real_path)
        fake_p = Path(spec.fake_path)

        if not real_p.is_absolute():
            real_p = PROJECT_ROOT / real_p
        if not fake_p.is_absolute():
            fake_p = PROJECT_ROOT / fake_p

        # Accept the spec if both real and fake paths exist
        if real_p.exists() and fake_p.exists():
            available.append(spec)
        else:
            # Try the alternative path pattern: datasets/<key> instead of datasets/test/<key>
            alt_real = PROJECT_ROOT / "datasets" / Path(spec.real_path).name
            alt_fake = PROJECT_ROOT / "datasets" / Path(spec.fake_path).name
            if alt_real.exists() and alt_fake.exists():
                available.append(TestDatasetSpec(
                    key=spec.key,
                    real_path=str(alt_real),
                    fake_path=str(alt_fake),
                    data_mode=spec.data_mode,
                ))

    return available


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_linear_model(arch, fc_path, device):
    """
    Instantiate model via get_model(opt), load fc weights from fc_path,
    and return the model in eval mode on device.
    """
    from models import get_model

    class _Opt:
        pass

    opt = _Opt()
    opt.arch = arch
    model = get_model(opt)

    state = torch.load(fc_path, map_location="cpu")
    # The checkpoint may contain just fc weights or a full state_dict
    if isinstance(state, dict) and any(k.startswith("fc.") for k in state):
        # Full-state-dict style: only load fc weights
        fc_state = {k[3:]: v for k, v in state.items() if k.startswith("fc.")}
        model.fc.load_state_dict(fc_state)
    elif isinstance(state, dict) and ("weight" in state or "bias" in state):
        # Bare fc layer state dict
        model.fc.load_state_dict(state)
    else:
        # Try loading as full model state dict
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # Fall back: assume it is a bare fc state dict
            model.fc.load_state_dict(state)

    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_linear_on_all(model, specs, arch, device,
                       max_sample=1000, jpeg_quality=None, gaussian_sigma=None):
    """
    For each spec: create TestDataset + DataLoader, call validate(),
    return {key: {"ap": float, "acc": float}}.
    """
    from data.test_data import TestDataset
    from utils.evaluation import validate

    results = {}
    model.eval()

    for spec in specs:
        try:
            ds = TestDataset(
                spec,
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
            )
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            ap, r_acc, f_acc, acc = validate(model, loader, device)
            results[spec.key] = {"ap": float(ap), "r_acc": float(r_acc),
                                  "f_acc": float(f_acc), "acc": float(acc)}
            print(f"    [{spec.key}] AP={ap:.4f}  acc={acc:.4f}")
        except Exception as e:
            print(f"    [{spec.key}] SKIPPED: {e}")

    return results


def eval_nn_on_all(model, bank, specs, arch, device, k=1, max_sample=1000):
    """
    For each spec: create TestDataset + DataLoader, call validate_nn(),
    return {key: {"ap": float, "acc": float}}.
    """
    from data.test_data import TestDataset
    from utils.nn_evaluation import validate_nn

    results = {}
    model.eval()

    for spec in specs:
        try:
            ds = TestDataset(spec, max_sample=max_sample, arch=arch)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            ap, r_acc, f_acc, acc = validate_nn(model, loader, bank, device, k=k)
            results[spec.key] = {"ap": float(ap), "r_acc": float(r_acc),
                                  "f_acc": float(f_acc), "acc": float(acc)}
            print(f"    [{spec.key}] AP={ap:.4f}  acc={acc:.4f}")
        except Exception as e:
            print(f"    [{spec.key}] SKIPPED: {e}")

    return results


# ---------------------------------------------------------------------------
# NN bank helper
# ---------------------------------------------------------------------------

def build_nn_bank(model, image_paths_real, image_paths_fake, arch, device, batch_size=128):
    """
    Build a FeatureBank from lists of real and fake image paths.
    Creates a simple inline dataset, extracts features, and returns FeatureBank.
    """
    import torchvision.transforms as T
    from torch.utils.data import Dataset as TorchDataset
    from models.nearest_neighbor import FeatureBank
    from utils.nn_evaluation import extract_features

    # Choose normalization stats based on arch
    if arch.lower().startswith("imagenet") or arch.lower().startswith("xception"):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std  = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    class _PathDataset(TorchDataset):
        def __init__(self, paths, labels):
            self.paths  = paths
            self.labels = labels

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img), self.labels[idx]

    all_paths  = image_paths_real + image_paths_fake
    all_labels = [0] * len(image_paths_real) + [1] * len(image_paths_fake)
    ds     = _PathDataset(all_paths, all_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    feats, labels = extract_features(model, loader, device)
    real_feats = feats[labels == 0].contiguous()
    fake_feats = feats[labels == 1].contiguous()
    return FeatureBank(real=real_feats, fake=fake_feats)


# ---------------------------------------------------------------------------
# Group AP aggregation
# ---------------------------------------------------------------------------

def group_ap(results):
    """
    Given {key: {"ap": float, ...}}, return {group: mean_ap} for
    GAN / Diffusion / Autoregressive based on DATASET_GROUPS.
    """
    group_vals = {g: [] for g in GROUP_ORDER}
    for key, metrics in results.items():
        group = DATASET_GROUPS.get(key)
        if group is not None:
            ap = metrics.get("ap", float("nan"))
            if not (isinstance(ap, float) and ap != ap):  # skip NaN
                group_vals[group].append(ap)

    return {
        g: float(np.mean(vals)) if vals else float("nan")
        for g, vals in group_vals.items()
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_train_opt(arch, data_path, data_label="train", batch_size=64):
    """
    Return an argparse.Namespace with all fields required by RealFakeDataset.
    """
    opt = argparse.Namespace(
        # data identity
        arch=arch,
        data_label=data_label,
        data_mode="wang2020",
        wang2020_data_path=str(data_path),
        isTrain=(data_label == "train"),
        # augmentation flags
        no_resize=False,
        no_crop=False,
        no_flip=False,
        # resize / crop
        rz_interp=["bilinear"],
        loadSize=256,
        cropSize=224,
        # blur + jpeg augmentation (matches paper's linear-head training setup)
        blur_prob=0.5,
        blur_sig=[0.0, 3.0],
        jpg_prob=0.5,
        jpg_method=["cv2", "pil"],
        jpg_qual=list(range(30, 101)),
        # list-mode paths (unused in wang2020 mode)
        real_list_path=None,
        fake_list_path=None,
        # dataloader settings
        serial_batches=False,
        class_bal=False,
        num_threads=0,
        batch_size=batch_size,
    )
    return opt


def train_linear_head(model, data_path, arch, device,
                      batch_size=64, lr=1e-4, max_epochs=30, patience=5,
                      max_sample_per_class=None, ckpt_path=None):
    """
    Train only the fc head of *model* (backbone frozen) on ProGAN-style
    wang2020 training data.  Returns model with best fc weights loaded.

    Args:
        model: nn.Module with a .fc attribute (Linear head).
        data_path: root path for wang2020 training data.
        arch: architecture string, e.g. 'CLIP:ViT-L/14'.
        device: torch.device.
        batch_size: batch size for training.
        lr: learning rate for fc head.
        max_epochs: maximum number of epochs.
        patience: early-stopping patience (epochs without val-loss improvement).
        max_sample_per_class: if set, truncate dataset to this many real/fake samples.
        ckpt_path: if set, save best fc weights here.

    Returns:
        model (eval mode, best fc weights loaded).
    """
    from data.dataset import RealFakeDataset

    # ---- Build train / val datasets ----------------------------------------
    train_opt = make_train_opt(arch, data_path, data_label="train", batch_size=batch_size)
    val_opt   = make_train_opt(arch, data_path, data_label="val",   batch_size=batch_size)
    val_opt.isTrain = False

    train_ds = RealFakeDataset(train_opt)
    val_ds   = RealFakeDataset(val_opt)

    # Optionally truncate dataset to max_sample_per_class
    if max_sample_per_class is not None:
        def _truncate(ds, n_per_class):
            real_idxs = [i for i, p in enumerate(ds.total_list) if ds.labels_dict[p] == 0]
            fake_idxs = [i for i, p in enumerate(ds.total_list) if ds.labels_dict[p] == 1]
            # Shuffle for reproducibility
            rng = np.random.default_rng(42)
            real_idxs = rng.permutation(real_idxs)[:n_per_class].tolist()
            fake_idxs = rng.permutation(fake_idxs)[:n_per_class].tolist()
            indices = real_idxs + fake_idxs
            return Subset(ds, indices)

        train_ds = _truncate(train_ds, max_sample_per_class)
        # Val: keep proportional cap (10% of max_sample_per_class, min 200)
        val_cap = max(200, max_sample_per_class // 10)
        val_ds  = _truncate(val_ds, val_cap)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Training fc head: {len(train_ds)} train, {len(val_ds)} val")

    # ---- Freeze backbone, only train fc ------------------------------------
    model.to(device)
    model.eval()  # keep backbone in eval (BN etc.)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.fc.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_fc_state = {k: v.clone() for k, v in model.fc.state_dict().items()}
    no_improve    = 0

    for epoch in range(1, max_epochs + 1):
        # --- train ---
        model.fc.train()
        train_loss = 0.0
        n_train    = 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            # Forward: get logits; handle models that may not accept return_feature kwarg
            with torch.no_grad():
                # Extract backbone features then pass through fc
                try:
                    feats = model(imgs, return_feature=True)
                except TypeError:
                    # Model like ImagenetModel doesn't accept return_feature
                    # We need to go through its backbone manually
                    feats = _extract_backbone_features(model, imgs)
            logits = model.fc(feats.float())
            loss   = criterion(logits.squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            n_train    += imgs.size(0)

        train_loss /= max(n_train, 1)

        # --- val ---
        model.fc.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().to(device)
                try:
                    feats = model(imgs, return_feature=True)
                except TypeError:
                    feats = _extract_backbone_features(model, imgs)
                logits = model.fc(feats.float())
                loss   = criterion(logits.squeeze(1), labels)
                val_loss += loss.item() * imgs.size(0)
                n_val    += imgs.size(0)

        val_loss /= max(n_val, 1)
        print(f"    Epoch {epoch:3d}/{max_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_fc_state = {k: v.clone() for k, v in model.fc.state_dict().items()}
            no_improve    = 0
            if ckpt_path:
                _save_fc(model.fc, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}.")
                break

    # Restore best fc weights
    model.fc.load_state_dict(best_fc_state)
    # Re-enable all gradients and set eval
    for param in model.parameters():
        param.requires_grad_(True)
    model.eval()
    return model


def _extract_backbone_features(model, imgs):
    """
    Extract penultimate-layer features from model backbone (before the fc head).
    Handles CLIPModel and ImagenetModel differently.
    """
    # CLIPModel: model.model.encode_image
    if hasattr(model, "model") and hasattr(model.model, "encode_image"):
        return model.model.encode_image(imgs).float()
    # ImagenetModel: model.model returns dict with "penultimate"
    if hasattr(model, "model"):
        out = model.model(imgs)
        if isinstance(out, dict) and "penultimate" in out:
            return out["penultimate"].float()
    # Generic fallback: run full forward and hope features come out
    raise RuntimeError(f"Cannot extract backbone features from {type(model)}")


def _save_fc(fc_layer, ckpt_path):
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fc_layer.state_dict(), str(path))
    print(f"    Checkpoint saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, float):
            return obj
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(_convert(data), f, indent=2)
    print(f"  Results saved to {path}")


def load_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure saving
# ---------------------------------------------------------------------------

def fig_save(fig, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    print(f"  Figure saved to {out_path}")
