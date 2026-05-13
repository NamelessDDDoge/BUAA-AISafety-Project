import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from data.test_data import TestDataset, TestDatasetSpec
from models.nearest_neighbor import FeatureBank, nearest_neighbor_scores
from utils.evaluation import calculate_acc, find_best_threshold
from utils.progress import print_batch_progress, should_report_progress


def extract_features(model, loader, device, progress_prefix=None):
    features, labels = [], []
    total_batches = len(loader)
    total_items = len(loader.dataset) if hasattr(loader, "dataset") else None
    seen = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(loader, start=1):
            in_tens = img.to(device)
            batch_features = model(in_tens, return_feature=True).detach().cpu()
            features.append(batch_features)
            labels.append(label.detach().cpu())
            seen += label.shape[0]
            if progress_prefix and should_report_progress(batch_idx, total_batches):
                print_batch_progress(
                    progress_prefix,
                    batch_idx,
                    total_batches,
                    seen,
                    total_items,
                )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).long()


def build_feature_bank(model, loader, device):
    features, labels = extract_features(
        model,
        loader,
        device,
        progress_prefix="Building NN feature bank",
    )
    real = features[labels == 0].contiguous()
    fake = features[labels == 1].contiguous()
    if real.numel() == 0 or fake.numel() == 0:
        raise ValueError("NN feature bank requires at least one real and one fake feature")
    return FeatureBank(real=real, fake=fake)


def load_or_build_feature_bank(model, opt, device):
    if opt.nn_bank_cache and os.path.exists(opt.nn_bank_cache):
        print(f"Loading NN feature bank cache: {opt.nn_bank_cache}", flush=True)
        bank_data = torch.load(opt.nn_bank_cache, map_location="cpu")
        return FeatureBank(real=bank_data["real"], fake=bank_data["fake"])

    real_path = opt.nn_bank_real_path or opt.nn_bank_path
    fake_path = opt.nn_bank_fake_path or opt.nn_bank_path
    spec = TestDatasetSpec(
        key="nn_bank",
        real_path=real_path,
        fake_path=fake_path,
        data_mode=opt.nn_bank_data_mode,
    )
    dataset = TestDataset(
        spec,
        max_sample=opt.nn_bank_max_sample,
        arch=opt.arch,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
    )
    print(f"Building NN feature bank from {len(dataset)} images...", flush=True)
    bank = build_feature_bank(model, loader, device)

    if opt.nn_bank_cache:
        cache_dir = os.path.dirname(opt.nn_bank_cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        print(f"Saving NN feature bank cache: {opt.nn_bank_cache}", flush=True)
        torch.save({"real": bank.real, "fake": bank.fake}, opt.nn_bank_cache)

    return bank


def validate_nn(
    model,
    loader,
    bank,
    device,
    k=1,
    bank_chunk_size=8192,
    find_thres=False,
    progress_prefix="NN validation",
):
    with torch.no_grad():
        y_true, y_pred = [], []
        total_batches = len(loader)
        total_items = len(loader.dataset) if hasattr(loader, "dataset") else None
        seen = 0
        print("Length of dataset: %d" % (total_batches), flush=True)
        for batch_idx, (img, label) in enumerate(loader, start=1):
            in_tens = img.to(device)
            features = model(in_tens, return_feature=True)
            scores = nearest_neighbor_scores(
                features,
                bank.real,
                bank.fake,
                k=k,
                bank_chunk_size=bank_chunk_size,
            )
            y_pred.extend(scores.detach().cpu().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            seen += label.shape[0]
            if should_report_progress(batch_idx, total_batches):
                print_batch_progress(
                    progress_prefix,
                    batch_idx,
                    total_batches,
                    seen,
                    total_items,
                )

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(f"{progress_prefix}: computing AP/accuracy metrics", flush=True)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)
    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
