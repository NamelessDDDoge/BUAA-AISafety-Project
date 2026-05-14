"""
Sec 6.7: Robustness to post-processing operations.
Tests CLIP:ViT-L/14 linear classifier under JPEG compression [100,90,80,70,60,50,40,30] and Gaussian blur sigmas [0,0.5,1,1.5,2,2.5,3,3.5,4]. Group datasets by family (GAN/Diffusion/Autoregressive). Plot: 2-subplot figure (left=JPEG, right=Gaussian), x=corruption level, y=mean group AP, 3 colored lines per subplot. No training needed.
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
    eval_linear_on_all, group_ap, save_json, fig_save
)

JPEG_QUALITIES = [100, 90, 80, 70, 60, 50, 40, 30]
GAUSSIAN_SIGMAS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    arch = 'CLIP:ViT-L/14'
    fc_path = Path(__file__).parent.parent / 'weights' / 'fc_weights.pth'
    model = load_linear_model(arch, str(fc_path), device)

    specs = get_available_specs()
    print(f"Available datasets: {[s.key for s in specs]}")

    # JPEG experiment
    jpeg_results = {}  # quality → group_ap
    for q in JPEG_QUALITIES:
        print(f"  JPEG quality={q}")
        res = eval_linear_on_all(model, specs, arch, device,
                                  max_sample=args.max_sample, jpeg_quality=q)
        jpeg_results[q] = group_ap(res)

    # Gaussian blur experiment
    blur_results = {}  # sigma → group_ap
    for s in GAUSSIAN_SIGMAS:
        print(f"  Gaussian sigma={s}")
        res = eval_linear_on_all(model, specs, arch, device,
                                  max_sample=args.max_sample, gaussian_sigma=s)
        blur_results[s] = group_ap(res)

    all_results = {'jpeg': jpeg_results, 'gaussian': blur_results}
    save_json(all_results, RESULTS_DIR / 'robustness.json')

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: JPEG
    ax = axes[0]
    for group in GROUP_ORDER:
        aps = [jpeg_results[q].get(group, float('nan')) for q in JPEG_QUALITIES]
        if not all(np.isnan(aps)):
            ax.plot(JPEG_QUALITIES, [a * 100 for a in aps],
                   marker='o', color=GROUP_COLORS[group], label=group, linewidth=2)
    ax.set_xlabel('JPEG Quality', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Robustness to JPEG Compression', fontsize=13, fontweight='bold')
    ax.invert_xaxis()
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Right: Gaussian blur
    ax = axes[1]
    for group in GROUP_ORDER:
        aps = [blur_results[s].get(group, float('nan')) for s in GAUSSIAN_SIGMAS]
        if not all(np.isnan(aps)):
            ax.plot(GAUSSIAN_SIGMAS, [a * 100 for a in aps],
                   marker='o', color=GROUP_COLORS[group], label=group, linewidth=2)
    ax.set_xlabel('Gaussian Blur σ', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title('Robustness to Gaussian Blur', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('CLIP:ViT-L/14 Linear Classifier – Robustness to Corruptions\n(trained on ProGAN)', fontsize=13)
    fig_save(fig, 'robustness')
    print("Done. Figure saved to experiments/results/robustness.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sample', type=int, default=500)
    args = parser.parse_args()
    run(args)
