# UniversalFakeDetect 复现说明

这个目录是在复现 WisconsinAIVision / UniversalFakeDetect：

https://github.com/WisconsinAIVision/UniversalFakeDetect#data

原项目论文为 **Towards Universal Fake Image Detectors that Generalize Across Generative Models**，CVPR 2023。核心目标是：用一种生成模型的数据训练出的检测器，泛化检测其他生成模型产生的假图像。

本仓库不是完整训练工程，而是当前机器上整理后的**测试/复现版本**：使用 CLIP ViT-L/14 图像主干，加上二分类头 `weights/fc_weights.pth`，跑 `test.py` 得到 AP 和 accuracy。

## 当前复现状态

已经跑通：

- `conda` 环境：`causal`
- 推理入口：`test.py`
- 模型结构：`CLIP:ViT-L/14`
- 二分类头：`weights/fc_weights.pth`
- CLIP 主干权重：通过原 OpenAI CLIP loader 自动下载/读取 cache 中的 `ViT-L/14.pt`
- 已验证数据：`datasets/progan`

注意：

- `weights/fc_weights.pth` 是必须使用的二分类头。
- `weights/clip.safetensors` 当前没有作为推理路径使用；实际跑通的是 OpenAI CLIP `.pt` 主干权重。
- 原始 README 里的 `validate.py` 在本仓库对应为 `test.py`。

## 环境

使用已有的 conda 环境：

```bash
conda run -n causal python test.py --help
```

如果环境里缺少 CLIP tokenizer 依赖，可以安装：

```bash
conda run -n causal python -m pip install ftfy
```

`torch`、`torchvision`、`safetensors` 等依赖在当前 `causal` 环境中已经可用。实际推理会自动选择 CUDA；如果没有 GPU，也会落到 CPU。

## 数据

原项目 Data 部分说明了两类测试数据：

1. Wang2020/CNNDetection 风格的 11 个 GAN/图像翻译测试集，例如 `progan`、`cyclegan`、`biggan` 等。
2. diffusion 模型测试集，例如 `guided`、`ldm_*`、`glide_*`、`dalle` 等，要求解压到 `./diffusion_datasets`。

当前仓库实际已有的数据在 `datasets/` 根目录下，而不是原 README 中的 `datasets/test/`。目录形态类似：

```text
datasets/
  progan/
    0_real/
    1_fake/
  cyclegan/
    0_real/
    1_fake/
  biggan/
    0_real/
    1_fake/
  ...
```

`data/datasets.yaml` 已经按当前目录写了这些路径。

当前缺失：

- 仓库根目录下没有 `diffusion_datasets/`。
- 因此如果直接不带 `--real_path/--fake_path` 跑全量配置，程序跑到 diffusion 数据集时会失败。
- 这不是代码问题，而是数据集缺失；需要按原项目说明下载并解压 diffusion 数据集。

## 运行单个数据集

推荐先跑单个已存在数据集，例如 `progan`：

```bash
conda run -n causal python test.py ^
  --arch CLIP:ViT-L/14 ^
  --ckpt weights/fc_weights.pth ^
  --real_path datasets/progan ^
  --fake_path datasets/progan ^
  --data_mode wang2020 ^
  --max_sample 1000 ^
  --batch_size 128
```

PowerShell 也可以写成一行：

```bash
conda run -n causal python test.py --arch CLIP:ViT-L/14 --ckpt weights/fc_weights.pth --real_path datasets/progan --fake_path datasets/progan --data_mode wang2020 --max_sample 1000 --batch_size 128
```

小样本 smoke test：

```bash
conda run -n causal python test.py --arch CLIP:ViT-L/14 --ckpt weights/fc_weights.pth --real_path datasets/progan --fake_path datasets/progan --data_mode wang2020 --max_sample 4 --batch_size 2
```

当前已验证该 smoke test 可以跑通，并在 `result/` 下生成：

```text
result/ap.txt
result/acc0.txt
```

## 跑配置文件中的数据集

`test.py` 在不传 `--real_path/--fake_path/--data_mode` 时，会读取：

```text
data/datasets.yaml
```

但当前 `data/datasets.yaml` 里同时包含了已有的 `datasets/*` 和缺失的 `diffusion_datasets/*`。所以现阶段更稳妥的方式是显式指定单个数据集路径。

如果之后补齐 `diffusion_datasets/`，可以尝试：

```bash
conda run -n causal python test.py --arch CLIP:ViT-L/14 --ckpt weights/fc_weights.pth --max_sample 1000 --batch_size 128
```

## 输出

结果保存在 `--result_folder` 指定目录，默认是：

```text
result/
```

主要文件：

- `ap.txt`：每个数据集的 Average Precision。
- `acc0.txt`：使用 0.5 阈值时的 real accuracy、fake accuracy、overall accuracy。

## 和原仓库的主要差异

- 原仓库入口叫 `validate.py`，当前仓库入口是 `test.py`。
- 原仓库默认数据路径示例是 `datasets/test/...`，当前数据直接放在 `datasets/...`。
- 当前只确认推理复现，不包含完整训练流程。
- `models.imagenet_models` 文件当前不存在，所以 Imagenet 分支不可用；CLIP 分支已通过懒加载方式避免被它阻塞。
- diffusion 测试数据当前缺失，需要另行下载。

## 引用

如果使用该项目，请引用原论文：

```bibtex
@inproceedings{ojha2023fakedetect,
      title={Towards Universal Fake Image Detectors that Generalize Across Generative Models},
      author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
      booktitle={CVPR},
      year={2023},
}
```
