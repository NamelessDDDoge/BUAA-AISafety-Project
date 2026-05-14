"""
Appendix C.1: Effect of different CLIP:ViT-L/14 layer choices.
Extracts features from transformer layers L0(idx=0), L8(idx=7), L16(idx=15), L24(final).
NN-based (k=1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from experiments.common import (
    RESULTS_DIR, GROUP_COLORS, GROUP_ORDER,
    get_available_specs, eval_nn_on_all,
    group_ap, save_json, fig_save
)
from models.clip_binary import CLIP_DOWNLOAD_ROOT
from models.clip import clip
from models.nearest_neighbor import FeatureBank
from utils.nn_evaluation import extract_features

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Paper's layer choices: L0, L8, L16, L24 (1-indexed, so block indices 0, 7, 15, 23)
# L24 = full CLIP output (no hook, use standard encode_image)
LAYER_CONFIGS = [
    ('L0',  0),   # output after 1st transformer block
    ('L8',  7),   # output after 8th transformer block
    ('L16', 15),  # output after 16th transformer block
    ('L24', None),# standard CLIP output (after ln_post + projection)
]

class CLIPLayerModel(nn.Module):
    """Wraps a raw CLIP model to extract features from a specific transformer layer."""

    def __init__(self, clip_raw, block_idx=None):
        """
        block_idx: int → extract CLS token after resblocks[block_idx] (0-indexed).
                   None → use standard encode_image output (768-dim projected).
        """
        super().__init__()
        self.clip_raw = clip_raw
        self.block_idx = block_idx
        self._hook_feat = None
        self.fc = nn.Linear(768, 1)  # dummy fc for compatibility

        if block_idx is not None:
            blocks = list(clip_raw.visual.transformer.resblocks.children())
            assert block_idx < len(blocks), f"block_idx {block_idx} >= {len(blocks)}"
            blocks[block_idx].register_forward_hook(self._capture_cls)

    def _capture_cls(self, module, inp, output):
        # output: (L, N, D) in LND format; output[0] = CLS token for all N samples
        self._hook_feat = output[0].detach().float()

    def forward(self, x, return_feature=True):
        self._hook_feat = None
        final_feat = self.clip_raw.encode_image(x).float()
        if self.block_idx is None:
            feat = final_feat
        else:
            feat = self._hook_feat
        if return_feature:
            return feat
        return self.fc(final_feat)

class SimpleImageDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths; self.labels = labels
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224), T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert('RGB')), self.labels[idx]

def collect_images(root, must_contain, max_n=None):
    from data.test_data import list_images
    imgs = list_images(root, must_contain=must_contain)
    if max_n: imgs = imgs[:max_n]
    return imgs

def build_bank(model, real_paths, fake_paths, device, batch_size=128):
    ds = SimpleImageDataset(real_paths + fake_paths, [0]*len(real_paths)+[1]*len(fake_paths))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = extract_features(model, loader, device)
    return FeatureBank(real=feats[labels==0].contiguous(), fake=feats[labels==1].contiguous())

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch = 'CLIP:ViT-L/14'
    print(f"Device: {device}")

    # Detect training data for building NN bank
    bank_path = args.bank_path
    if not bank_path:
        for candidate in ['datasets/train', 'datasets']:
            p = Path(__file__).parent.parent / candidate
            if (p / 'progan').exists():
                bank_path = str(p / 'progan')
                break
            elif p.exists():
                bank_path = str(p)
                break

    if not bank_path or not Path(bank_path).exists():
        print("ERROR: NN bank data not found. Pass --bank_path <path>")
        return

    print(f"Bank data path: {bank_path}")
    real_paths = collect_images(bank_path, '0_real', args.max_bank)
    fake_paths = collect_images(bank_path, '1_fake', args.max_bank)
    print(f"Bank: {len(real_paths)} real, {len(fake_paths)} fake")

    if not real_paths or not fake_paths:
        print("ERROR: No images found in bank path with 0_real/1_fake structure.")
        return

    specs = get_available_specs()
    print(f"Available test datasets: {[s.key for s in specs]}")

    # Load base CLIP model once (shared weights, different hooks)
    print("\nLoading CLIP:ViT-L/14...")
    clip_raw, _ = clip.load('ViT-L/14', device='cpu', download_root=str(CLIP_DOWNLOAD_ROOT))

    all_results = {}

    for layer_label, block_idx in LAYER_CONFIGS:
        print(f"\n=== Layer {layer_label} (block_idx={block_idx}) ===")
        model = CLIPLayerModel(clip_raw, block_idx=block_idx)
        model.eval()
        model.to(device)

        print(f"  Building NN feature bank...")
        bank = build_bank(model, real_paths, fake_paths, device, batch_size=args.batch_size)
        print(f"  Bank: real={bank.real.shape}, fake={bank.fake.shape}")

        print(f"  Evaluating...")
        res = eval_nn_on_all(model, bank, specs, arch, device, k=1, max_sample=args.max_sample)
        all_results[layer_label] = {'block_idx': block_idx, 'per_dataset': res, 'per_group': group_ap(res)}

    save_json(all_results, RESULTS_DIR / 'clip_layers.json')

    if not all_results:
        print("No results to plot.")
        return

    # Plot: line chart, x=layer label, y=AP per group
    layer_labels = list(all_results.keys())
    x = np.arange(len(layer_labels))

    fig, ax = plt.subplots(figsize=(8, 5))

    for group in GROUP_ORDER:
        aps = [all_results[l]['per_group'].get(group, float('nan')) * 100 for l in layer_labels]
        if not all(np.isnan(aps)):
            ax.plot(x, aps, marker='o', color=GROUP_COLORS[group],
                   label=group, linewidth=2, markersize=8)
            for xi, ap in zip(x, aps):
                if not np.isnan(ap):
                    ax.annotate(f'{ap:.1f}', (xi, ap), textcoords='offset points',
                               xytext=(0, 8), ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=11)
    ax.set_xlabel('CLIP:ViT-L/14 Layer', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Effect of CLIP Layer Choice on NN Classification\n(k=1, ProGAN training bank)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Add annotation explaining layers
    ax.text(0.02, 0.05, 'L0 ≈ pixel space\nL24 = full CLIP output',
            transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

    fig_save(fig, 'clip_layers')
    print("\nDone. Figure saved to experiments/results/clip_layers.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank_path', type=str, default=None, help='Path to ProGAN training data root for building NN bank')
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--max_bank', type=int, default=5000, help='Max images per class for NN bank')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    run(args)
