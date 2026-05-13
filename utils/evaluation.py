from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score


def find_best_threshold(y_true, y_pred):
    real_scores = y_pred[y_true == 0]
    fake_scores = y_pred[y_true == 1]

    if real_scores.size == 0 or fake_scores.size == 0:
        raise ValueError("find_best_threshold requires both real and fake labels")
    if real_scores.max() <= fake_scores.min():
        return (real_scores.max() + fake_scores.min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / y_true.shape[0]
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(trainer, loader, device, find_thres=False):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in loader:
            in_tens = img.to(device)

            output = model(in_tens)

            if len(output.shape) == 4:
                probs = torch.softmax(output, dim=1)
                scores = torch.mean(probs[:, 1, :, :], dim=(1, 2))
            else:
                scores = output.sigmoid().flatten()

            y_pred.extend(scores.tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ap = average_precision_score(y_true, y_pred)
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0

    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
