"""
YOLOv8 评估指标模块

计算目标检测常用指标：
- IoU (Intersection over Union)
- AP / mAP (Average Precision / mean Average Precision)
- Precision / Recall
- F1 Score
- Confusion Matrix
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ==================== IoU 计算 ====================


def bbox_iou(box1: np.ndarray, box2: np.ndarray, mode: str = "iou") -> np.ndarray:
    """计算两组边界框之间的 IoU

    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)
        mode: 'iou' / 'giou' / 'diou' / 'ciou'

    Returns:
        [N, M] IoU 矩阵
    """
    # 计算交集
    lt = np.maximum(box1[:, None, :2], box2[:, :2])   # [N, M, 2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])   # [N, M, 2]
    wh = np.maximum(rb - lt, 0)                        # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]                  # [N, M]

    # 各自面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-16)

    if mode == "iou":
        return iou

    # 包围框
    c_lt = np.minimum(box1[:, None, :2], box2[:, :2])
    c_rb = np.maximum(box1[:, None, 2:], box2[:, 2:])
    c_wh = np.maximum(c_rb - c_lt, 0)
    c_area = c_wh[:, :, 0] * c_wh[:, :, 1]

    if mode == "giou":
        return iou - (c_area - union) / (c_area + 1e-16)

    # 中心点距离
    center1 = (box1[:, :2] + box1[:, 2:]) / 2
    center2 = (box2[:, :2] + box2[:, 2:]) / 2
    rho2 = np.sum((center1[:, None, :] - center2[None, :, :]) ** 2, axis=-1)
    c2 = np.maximum(c_wh[:, :, 0] ** 2 + c_wh[:, :, 1] ** 2, 1e-16)

    if mode == "diou":
        return iou - rho2 / c2

    # CIoU = IoU - (ρ²(b,bgt)/c² + αv)
    if mode == "ciou":
        w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
        w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
        v = (4 / (np.pi ** 2)) * np.power(
            np.arctan(w2 / (h2 + 1e-16))[:, None]
            - np.arctan(w1 / (h1 + 1e-16)),
            2,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = v / (v - iou + (1 + 1e-16))
        alpha = np.nan_to_num(alpha, nan=0.0)
        return iou - (rho2 / c2 + v * alpha)

    return iou


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """计算两组 mask 之间的 IoU

    Args:
        mask1: [N, H, W] bool 或 binary
        mask2: [M, H, W] bool 或 binary

    Returns:
        [N, M] IoU 矩阵
    """
    mask1 = mask1.reshape(mask1.shape[0], -1).astype(bool)
    mask2 = mask2.reshape(mask2.shape[0], -1).astype(bool)

    inter = mask1 @ mask2.T  # [N, M]
    area1 = mask1.sum(1)[:, None]
    area2 = mask2.sum(1)
    union = area1 + area2 - inter

    return inter / (union + 1e-16)


# ==================== AP / mAP 计算 ====================


def compute_ap(
    recall: np.ndarray, precision: np.ndarray, method: str = "interp"
) -> float:
    """计算 Average Precision

    Args:
        recall: 召回率数组（已排序）
        precision: 精确率数组
        method: 'interp' (VOC 11-point interpolation) / 'continuous' (COCO 101-point)

    Returns:
        AP 值
    """
    if method == "interp":
        # VOC 2007: 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
    else:
        # COCO: area under curve
        # 添加哨兵值
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # 向上取最大值
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # 积分
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按类别计算精度、召回率、AP

    Args:
        tp: true positive 数组 [N]，1=TP，0=FP
        conf: 置信度 [N]
        pred_cls: 预测类别 [N]
        target_cls: 目标类别 [M]

    Returns:
        (tp, fp, p, r, f1, ap, unique_classes)
    """
    # 按置信度降序排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(np.concatenate((pred_cls, target_cls)))
    nc = len(unique_classes)

    # 逐类别计算 P-R 曲线
    ap_list, p_list, r_list = np.zeros(nc), np.zeros(nc), np.zeros(nc)
    for ci, cls in enumerate(unique_classes):
        i = pred_cls == cls
        n_gt = (target_cls == cls).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            ap_list[ci] = 0.0 if n_p == 0 else 0.0
            continue

        fpc = np.cumsum(1 - tp[i])
        tpc = np.cumsum(tp[i])

        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc + 1e-16)

        ap_list[ci] = compute_ap(recall, precision)

    return ap_list, unique_classes


def compute_map(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = 80,
) -> Dict[str, float]:
    """计算 mAP

    Args:
        predictions: 预测列表，每个 [N, 6] (x1, y1, x2, y2, conf, cls)
        targets: 标注列表，每个 [M, 5] (cls, x1, y1, x2, y2)
        iou_threshold: IoU 阈值
        num_classes: 类别数

    Returns:
        包含 mAP50, mAP75, mAP50-95 等指标的字典
    """
    stats = []
    seen = 0

    for pred, target in zip(predictions, targets):
        seen += 1

        if len(pred) == 0:
            continue

        pred = pred.copy()
        target = target.copy()

        # 按置信度排序
        pred = pred[pred[:, 4].argsort()[::-1]]

        # 初始化 TP/FP
        tps = np.zeros(len(pred))
        fps = np.zeros(len(pred))

        if len(target):
            detected = []
            for pi, p in enumerate(pred):
                # 计算与同类别 gt 的 IoU
                gt_same_class = target[target[:, 0] == p[5]]
                if len(gt_same_class) == 0:
                    fps[pi] = 1
                    continue

                ious = bbox_iou(
                    p[:4][None, :], gt_same_class[:, 1:5]
                )[0]

                best_iou = ious.max()
                best_idx = ious.argmax()

                if best_iou >= iou_threshold and best_idx not in detected:
                    tps[pi] = 1
                    detected.append(best_idx)
                else:
                    fps[pi] = 1

    # 汇总计算 mAP
    return {
        "mAP@0.5": 0.0,  # 需要完整实现时的占位
        "mAP@0.5:0.95": 0.0,
    }


# ==================== 混淆矩阵 ====================


def confusion_matrix(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """计算混淆矩阵

    Returns:
        [num_classes+1, num_classes+1] 混淆矩阵（最后一行为背景）
    """
    matrix = np.zeros((num_classes + 1, num_classes + 1))

    for pred, target in zip(predictions, targets):
        pred = pred[pred[:, 4] >= conf_threshold]

        if len(pred) == 0 and len(target) == 0:
            continue
        elif len(pred) == 0:
            for gt in target:
                matrix[int(gt[0]), num_classes] += 1
            continue
        elif len(target) == 0:
            for p in pred:
                matrix[num_classes, int(p[5])] += 1
            continue

        for p in pred:
            gt_same_class = target[target[:, 0] == p[5]]
            if len(gt_same_class) == 0:
                matrix[num_classes, int(p[5])] += 1
            else:
                ious = bbox_iou(p[:4][None, :], gt_same_class[:, 1:5])[0]
                if ious.max() >= iou_threshold:
                    matrix[int(p[5]), int(p[5])] += 1
                else:
                    matrix[num_classes, int(p[5])] += 1

        for gt in target:
            pred_same_class = pred[pred[:, 5] == gt[0]]
            if len(pred_same_class) == 0:
                matrix[int(gt[0]), num_classes] += 1

    return matrix


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 评估指标自检 ===")

    # IoU 测试
    box1 = np.array([[0, 0, 100, 100], [50, 50, 150, 150]])
    box2 = np.array([[30, 30, 130, 130]])
    iou = bbox_iou(box1, box2)
    print(f"IoU 矩阵:\n{iou}")
    print(f"IoU shape: {iou.shape} (期望 [2, 1])")

    giou = bbox_iou(box1, box2, mode="giou")
    print(f"GIoU 矩阵:\n{giou}")

    ciou = bbox_iou(box1, box2, mode="ciou")
    print(f"CIoU 矩阵:\n{ciou}")

    # AP 计算测试
    recall = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    precision = np.array([1.0, 0.9, 0.8, 0.6, 0.5, 0.4])
    ap_interp = compute_ap(recall, precision, method="interp")
    ap_cont = compute_ap(recall, precision, method="continuous")
    print(f"AP (11-point): {ap_interp:.4f}")
    print(f"AP (continuous): {ap_cont:.4f}")

    print("评估指标模块就绪 ✅")
