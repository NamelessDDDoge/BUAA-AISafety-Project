"""
Appendix C.2: Effect of dataset diversity on NN generalization.
Varies number of LSUN ProGAN training classes: 2, 4, 8, 20.
NN-based (no fc training needed).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from experiments.common import (
    RESULTS_DIR, GROUP_COLORS, GROUP_ORDER,
    get_available_specs, eval_nn_on_all,
    group_ap, save_json, fig_save
)
from models.clip_binary import CLIPModel, CLIP_DOWNLOAD_ROOT
from models.clip import clip
from models.nearest_neighbor import FeatureBank
from utils.nn_evaluation import extract_features

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
NUM_CLASSES_LIST = [2, 4, 8, 20]

def find_images_by_class(root, must_contain):
    """Returns dict {class_name: [img_paths]}."""
    root = Path(root)
    class_images = {}
    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)
        if must_contain not in str(dirpath):
            continue
        imgs = [str(dirpath / f) for f in filenames if Path(f).suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        # Determine class name: component between root and must_contain folder
        try:
            rel = dirpath.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        # The class is whichever part is NOT must_contain
        class_name = next((p for p in parts if p != must_contain.strip('/')), str(rel))
        class_images.setdefault(class_name, []).extend(imgs)
    return class_images

class SimpleImageDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

def build_bank_from_paths(model, real_paths, fake_paths, device, batch_size=128):
    ds = SimpleImageDataset(real_paths + fake_paths, [0]*len(real_paths) + [1]*len(fake_paths))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = extract_features(model, loader, device)
    return FeatureBank(real=feats[labels==0].contiguous(), fake=feats[labels==1].contiguous())

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch = 'CLIP:ViT-L/14'
    print(f"Device: {device}")

    # Load CLIP model (backbone only, no fc needed for NN)
    clip_model_raw, _ = clip.load('ViT-L/14', device='cpu', download_root=str(CLIP_DOWNLOAD_ROOT))

    class _CLIPFeatureModel(torch.nn.Module):
        def __init__(self, clip_raw):
            super().__init__()
            self.model = clip_raw
            self.fc = torch.nn.Linear(768, 1)  # dummy fc
        def forward(self, x, return_feature=True):
            feats = self.model.encode_image(x).float()
            if return_feature:
                return feats
            return self.fc(feats)

    model = _CLIPFeatureModel(clip_model_raw)
    model.eval()
    model.to(device)

    # Detect training data
    data_path = args.data_path
    if not data_path:
        for candidate in ['datasets/train', 'datasets']:
            p = Path(__file__).parent.parent / candidate
            for subdir in ['progan', '.']:
                pp = p / subdir
                if pp.exists():
                    # Check for 0_real subdir somewhere
                    has_real = any('0_real' in str(x) for x in pp.rglob('*') if x.is_dir())
                    if has_real:
                        data_path = str(pp)
                        break
            if data_path:
                break

    if not data_path or not Path(data_path).exists():
        print("ERROR: Training data not found. Pass --data_path <path_to_progan_root>")
        print("Expected structure: <path>/class1/0_real/ and <path>/class1/1_fake/")
        return

    print(f"Training data: {data_path}")

    # Discover classes
    real_by_class = find_images_by_class(data_path, '0_real')
    fake_by_class = find_images_by_class(data_path, '1_fake')
    common_classes = sorted(set(real_by_class.keys()) & set(fake_by_class.keys()))
    print(f"Found {len(common_classes)} classes: {common_classes[:5]}{'...' if len(common_classes)>5 else ''}")

    if len(common_classes) == 0:
        print("ERROR: No class structure found. Are images organized by class?")
        return

    specs = get_available_specs()
    print(f"Available test datasets: {[s.key for s in specs]}")

    all_results = {}

    for n_classes in NUM_CLASSES_LIST:
        if n_classes > len(common_classes):
            print(f"  [skip] n_classes={n_classes} > available {len(common_classes)}")
            continue
        selected = common_classes[:n_classes]
        print(f"\n=== {n_classes} classes: {selected} ===")

        real_paths = []
        fake_paths = []
        for cls in selected:
            reals = real_by_class.get(cls, [])
            fakes = fake_by_class.get(cls, [])
            if args.max_bank_per_class:
                reals = reals[:args.max_bank_per_class]
                fakes = fakes[:args.max_bank_per_class]
            real_paths.extend(reals)
            fake_paths.extend(fakes)

        print(f"  Bank: {len(real_paths)} real, {len(fake_paths)} fake")
        bank = build_bank_from_paths(model, real_paths, fake_paths, device)

        res = eval_nn_on_all(model, bank, specs, arch, device, k=1, max_sample=args.max_sample)
        all_results[str(n_classes)] = {'per_dataset': res, 'per_group': group_ap(res)}

    save_json(all_results, RESULTS_DIR / 'diversity.json')

    if not all_results:
        print("No results to plot.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    n_vals = [int(k) for k in all_results.keys()]

    for group in GROUP_ORDER:
        aps = [all_results[str(n)]['per_group'].get(group, float('nan')) * 100 for n in n_vals]
        if not all(np.isnan(aps)):
            ax.plot(n_vals, aps, marker='o', color=GROUP_COLORS[group],
                   label=group, linewidth=2, markersize=8)

    ax.set_xlabel('Number of Training Classes (LSUN Categories)', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Effect of Dataset Diversity\n(CLIP:ViT-L/14 NN, k=1, ProGAN training data)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xticks(n_vals)
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig_save(fig, 'diversity')
    print("\nDone. Figure saved to experiments/results/diversity.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Path to ProGAN training data root (containing class subfolders)')
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--max_bank_per_class', type=int, default=2000)
    args = parser.parse_args()
    run(args)
