"""
Sec 6.3: Effect of network backbone and pre-training dataset.
Compares CLIP:ViT-L/14, CLIP:RN50, Imagenet:resnet50, Imagenet:vit_b_16.
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

from experiments.common import (
    RESULTS_DIR, GROUP_COLORS, GROUP_ORDER,
    get_available_specs, load_linear_model,
    eval_linear_on_all, group_ap, train_linear_head,
    save_json, fig_save
)
from models import get_model

BACKBONES = [
    ('CLIP:ViT-L/14', 'CLIP ViT-L/14\n(CLIP pretrain)'),
    ('CLIP:RN50',     'CLIP ResNet-50\n(CLIP pretrain)'),
    ('Imagenet:resnet50', 'ResNet-50\n(ImageNet pretrain)'),
    ('Imagenet:vit_b_16', 'ViT-B/16\n(ImageNet pretrain)'),
]

PRETRAINED_FC = {
    'CLIP:ViT-L/14': 'weights/fc_weights.pth',
}

def get_ckpt_path(arch):
    safe = arch.replace('/', '_').replace(':', '_')
    return RESULTS_DIR / 'backbone_ckpts' / f'{safe}_fc.pth'

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    specs = get_available_specs()
    print(f"Available datasets: {[s.key for s in specs]}")

    all_results = {}

    for arch, label in BACKBONES:
        print(f"\n=== Backbone: {arch} ===")
        ckpt = get_ckpt_path(arch)
        pretrained = PRETRAINED_FC.get(arch)

        # Determine fc weights path
        if pretrained and (Path(pretrained).exists() or (Path(__file__).parent.parent / pretrained).exists()):
            fc_path = pretrained if Path(pretrained).exists() else str(Path(__file__).parent.parent / pretrained)
            print(f"  Using pretrained fc: {fc_path}")
            model = load_linear_model(arch, fc_path, device)
        elif ckpt.exists():
            print(f"  Loading existing checkpoint: {ckpt}")
            model = load_linear_model(arch, str(ckpt), device)
        else:
            # Need to train
            data_path = args.data_path
            if not data_path:
                # Try to detect
                for candidate in ['datasets/train', 'datasets']:
                    p = Path(__file__).parent.parent / candidate
                    if p.exists():
                        data_path = str(p)
                        break
            if not data_path or not Path(data_path).exists():
                print(f"  [skip] No training data found. Pass --data_path. Skipping {arch}")
                continue

            class _Opt:
                pass
            opt = _Opt()
            opt.arch = arch
            model = get_model(opt)
            model = train_linear_head(
                model, data_path, arch, device,
                batch_size=args.batch_size,
                lr=1e-4, max_epochs=args.epochs, patience=5,
                ckpt_path=str(ckpt)
            )

        print(f"  Evaluating {arch}...")
        res = eval_linear_on_all(model, specs, arch, device, max_sample=args.max_sample)
        all_results[arch] = {'label': label, 'per_dataset': res, 'per_group': group_ap(res)}

    save_json(all_results, RESULTS_DIR / 'backbone.json')

    if not all_results:
        print("No results. Exiting.")
        return

    # Plot: grouped bar chart
    arch_keys = list(all_results.keys())
    arch_labels = [all_results[a]['label'] for a in arch_keys]
    x = np.arange(len(arch_keys))
    n_groups = len(GROUP_ORDER)
    bar_width = 0.25
    offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * bar_width

    fig, ax = plt.subplots(figsize=(max(8, len(arch_keys) * 2.5), 6))

    for i, group in enumerate(GROUP_ORDER):
        aps = [all_results[a]['per_group'].get(group, float('nan')) * 100 for a in arch_keys]
        bars = ax.bar(x + offsets[i], aps, width=bar_width,
                      color=GROUP_COLORS[group], label=group, alpha=0.85, edgecolor='white')
        for bar, ap in zip(bars, aps):
            if not np.isnan(ap):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{ap:.1f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Chance (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=10)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Effect of Network Backbone and Pre-training Dataset\n(linear probing, trained on ProGAN)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig_save(fig, 'backbone')
    print("\nDone. Figure saved to experiments/results/backbone.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data (wang2020 format root)')
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    run(args)
