"""Create a polished SVG for the completed JPEG robustness experiment."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "result" / "log.txt"
OUT_DIR = PROJECT_ROOT / "experiments" / "results"
SVG_PATH = OUT_DIR / "robustness_jpeg_cn.svg"
PNG_PATH = OUT_DIR / "robustness_jpeg_cn_preview.png"

JPEG_QUALITIES = [100, 90, 80, 70, 60, 50, 40, 30]
DATASET_GROUPS = {
    "progan": "GAN",
    "cyclegan": "GAN",
    "biggan": "GAN",
    "stylegan": "GAN",
    "gaugan": "GAN",
    "stargan": "GAN",
    "deepfake": "GAN",
    "sitd": "GAN",
    "san": "GAN",
    "crn": "GAN",
    "imle": "GAN",
    "ldm_200": "Diffusion",
    "ldm_200_cfg": "Diffusion",
    "ldm_100": "Diffusion",
    "glide_100_27": "Diffusion",
    "glide_50_27": "Diffusion",
    "glide_100_10": "Diffusion",
    "guided": "Diffusion",
    "dalle": "Autoregressive",
}
GROUP_ORDER = ["GAN", "Diffusion", "Autoregressive"]
GROUP_LABELS = {
    "GAN": "GAN 生成模型",
    "Diffusion": "扩散模型",
    "Autoregressive": "自回归模型",
}
GROUP_COLORS = {
    "GAN": "#3B4CC0",
    "Diffusion": "#8DB0FE",
    "Autoregressive": "#B40426",
}
GROUP_MARKERS = {
    "GAN": "o",
    "Diffusion": "s",
    "Autoregressive": "^",
}


def parse_jpeg_log(log_path: Path) -> dict[int, dict[str, float]]:
    quality_pattern = re.compile(r"JPEG quality=(\d+)")
    metric_pattern = re.compile(r"\s+\[([^\]]+)\]\s+AP=([0-9.]+)")

    current_quality: int | None = None
    parsed: dict[int, dict[str, float]] = defaultdict(dict)

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        quality_match = quality_pattern.search(line)
        if quality_match:
            current_quality = int(quality_match.group(1))
            continue

        metric_match = metric_pattern.search(line)
        if metric_match and current_quality is not None:
            dataset, ap = metric_match.groups()
            parsed[current_quality][dataset] = float(ap)

    missing = [quality for quality in JPEG_QUALITIES if quality not in parsed]
    if missing:
        raise ValueError(f"Missing JPEG qualities in log: {missing}")

    return dict(parsed)


def aggregate_groups(results: dict[int, dict[str, float]]) -> dict[str, list[float]]:
    aggregated: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}

    for quality in JPEG_QUALITIES:
        group_values: dict[str, list[float]] = {group: [] for group in GROUP_ORDER}
        for dataset, ap in results[quality].items():
            group = DATASET_GROUPS.get(dataset)
            if group is not None:
                group_values[group].append(ap * 100)

        for group in GROUP_ORDER:
            values = group_values[group]
            aggregated[group].append(float(np.mean(values)) if values else np.nan)

    return aggregated


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": ["STZhongSong", "Book Antiqua", "serif"],
            "font.serif": ["STZhongSong", "Book Antiqua", "Times New Roman"],
            "font.sans-serif": ["STZhongSong", "Book Antiqua", "Arial"],
            "axes.edgecolor": "#555555",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_terminal_guides(ax: plt.Axes, group_values: dict[str, list[float]]) -> None:
    label_y = {
        "GAN": group_values["GAN"][-1] - 0.9,
        "Diffusion": group_values["Diffusion"][-1] + 0.2,
        "Autoregressive": group_values["Autoregressive"][-1] + 0.8,
    }
    for group in GROUP_ORDER:
        last = group_values[group][-1]
        ax.hlines(
            y=last,
            xmin=JPEG_QUALITIES[-1],
            xmax=104,
            color=GROUP_COLORS[group],
            linestyle=(0, (3, 3)),
            linewidth=1.1,
            alpha=0.7,
            zorder=0,
        )
        ax.annotate(
            f"{last:.1f}",
            xy=(104, last),
            xytext=(105.45, label_y[group]),
            ha="right",
            va="center",
            color=GROUP_COLORS[group],
            fontsize=9.5,
            arrowprops={
                "arrowstyle": "-",
                "color": GROUP_COLORS[group],
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )


def set_terminal_axis_ticks(ax: plt.Axes, group_values: dict[str, list[float]]) -> None:
    ax.set_yticks([50, 60, 70, 80, 90, 100])


def make_figure(group_values: dict[str, list[float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    for group in GROUP_ORDER:
        values = group_values[group]
        ax.plot(
            JPEG_QUALITIES,
            values,
            label=GROUP_LABELS[group],
            color=GROUP_COLORS[group],
            marker=GROUP_MARKERS[group],
            markersize=6,
            linewidth=2.4,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )

    add_terminal_guides(ax, group_values)

    ax.axhline(
        50,
        color="#9A9A9A",
        linestyle=(0, (4, 4)),
        linewidth=1.1,
        label="随机基线 50%",
        zorder=0,
    )
    ax.fill_between(JPEG_QUALITIES, 50, 100, color="#EAF2F8", alpha=0.25, zorder=-2)

    ax.set_title("JPEG 压缩鲁棒性：CLIP:ViT-L/14 线性分类器", pad=12, fontsize=13.5)
    ax.set_xlabel("JPEG 质量（越低表示压缩越强）", labelpad=8)
    ax.set_ylabel("平均 AP（%）", labelpad=8)
    ax.set_xticks(JPEG_QUALITIES)
    ax.invert_xaxis()
    ax.set_ylim(45, 101.5)
    ax.set_xlim(104, 24)
    set_terminal_axis_ticks(ax, group_values)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.012, 0.045),
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#D0D0D0",
        fontsize=10.5,
    )
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    return fig


def main() -> None:
    configure_matplotlib()
    results = parse_jpeg_log(LOG_PATH)
    group_values = aggregate_groups(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure(group_values)
    fig.savefig(SVG_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for group in GROUP_ORDER:
        values = group_values[group]
        print(
            f"{group}: q100={values[0]:.2f}, q30={values[-1]:.2f}, "
            f"drop={values[0] - values[-1]:.2f}"
        )
    print(f"SVG: {SVG_PATH}")
    print(f"Preview: {PNG_PATH}")


if __name__ == "__main__":
    main()
