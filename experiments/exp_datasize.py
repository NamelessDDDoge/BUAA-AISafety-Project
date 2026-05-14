"""
Sec 6.5: Effect of training data size.
Trains CLIP:ViT-L/14 linear head with different amounts of ProGAN data.
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

SIZES_TOTAL = [8000, 20000, 80000, 200000, 720000]  # total real+fake images

def get_ckpt_path(size):
    return RESULTS_DIR / 'datasize_ckpts' / f'size_{size}_fc.pth'

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arch = 'CLIP:ViT-L/14'
    print(f"Device: {device}")

    # Detect training data
    data_path = args.data_path
    if not data_path:
        for candidate in ['datasets/train', 'datasets']:
            p = Path(__file__).parent.parent / candidate
            if p.exists():
                data_path = str(p)
                break
    if not data_path or not Path(data_path).exists():
        print("ERROR: Training data not found. Pass --data_path <path_to_wang2020_root>")
        return

    specs = get_available_specs()
    print(f"Available datasets: {[s.key for s in specs]}")

    all_results = {}

    for size in SIZES_TOTAL:
        per_class = size // 2
        print(f"\n=== Training size: {size} (≈{per_class} per class) ===")
        ckpt = get_ckpt_path(size)

        if ckpt.exists() and not args.retrain:
            print(f"  Loading existing checkpoint: {ckpt}")
            model = load_linear_model(arch, str(ckpt), device)
        else:
            class _Opt:
                pass
            opt = _Opt()
            opt.arch = arch
            model = get_model(opt)
            model = train_linear_head(
                model, data_path, arch, device,
                batch_size=args.batch_size, lr=1e-4,
                max_epochs=args.epochs, patience=5,
                max_sample_per_class=per_class,
                ckpt_path=str(ckpt)
            )

        print(f"  Evaluating size={size}...")
        res = eval_linear_on_all(model, specs, arch, device, max_sample=args.max_sample)
        all_results[str(size)] = {'per_dataset': res, 'per_group': group_ap(res)}

    save_json(all_results, RESULTS_DIR / 'datasize.json')

    if not all_results:
        print("No results to plot.")
        return

    # Plot: line chart with log x-axis
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = [int(s) for s in all_results.keys()]
    for group in GROUP_ORDER:
        aps = [all_results[str(s)]['per_group'].get(group, float('nan')) * 100 for s in sizes]
        if not all(np.isnan(aps)):
            ax.plot(sizes, aps, marker='o', color=GROUP_COLORS[group],
                   label=group, linewidth=2, markersize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (real + fake)', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Effect of Training Data Size\n(CLIP:ViT-L/14 linear probing, ProGAN training data)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s//1000}k' if s >= 1000 else str(s) for s in sizes], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig_save(fig, 'datasize')
    print("\nDone. Figure saved to experiments/results/datasize.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    run(args)
