from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FeatureBank:
    real: torch.Tensor
    fake: torch.Tensor


def normalize_features(features):
    # 将特征向量归一化为单位长度，便于后续计算余弦相似度。
    return F.normalize(features.float(), dim=1)


def topk_cosine_distances(query_features, bank_features, k=1, bank_chunk_size=8192):
    # 计算每个查询特征到特征库中 k 个最近邻的平均余弦距离。
    if k < 1:
        raise ValueError("k must be >= 1")
    if bank_features.ndim != 2:
        raise ValueError("bank_features must be a 2D tensor")
    if query_features.ndim != 2:
        raise ValueError("query_features must be a 2D tensor")
    if bank_features.shape[0] == 0:
        raise ValueError("feature bank is empty")

    k = min(k, bank_features.shape[0])
    query_features = normalize_features(query_features)
    best = None

    for start in range(0, bank_features.shape[0], bank_chunk_size):
        chunk = bank_features[start : start + bank_chunk_size].to(query_features.device)
        chunk = normalize_features(chunk)
        sims = query_features @ chunk.t()
        chunk_best = sims.topk(k=min(k, sims.shape[1]), dim=1).values
        best = chunk_best if best is None else torch.cat([best, chunk_best], dim=1)
        if best.shape[1] > k:
            best = best.topk(k=k, dim=1).values

    return 1.0 - best.mean(dim=1)


def nearest_neighbor_scores(
    query_features,
    real_bank,
    fake_bank,
    k=1,
    bank_chunk_size=8192,
    eps=1e-12,
):
    # 根据查询特征到真实库和伪造库的相对距离，得到伪造置信分数。
    real_dist = topk_cosine_distances(
        query_features, real_bank, k=k, bank_chunk_size=bank_chunk_size
    )
    fake_dist = topk_cosine_distances(
        query_features, fake_bank, k=k, bank_chunk_size=bank_chunk_size
    )
    return real_dist / (real_dist + fake_dist + eps)


def nearest_neighbor_predictions(scores, threshold=0.5):
    # 将伪造置信分数转换为二分类标签：0 表示真实，1 表示伪造。
    return (scores > threshold).long()
