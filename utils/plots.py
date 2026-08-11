"""
YOLOv8 可视化模块

提供检测结果、训练曲线、混淆矩阵等可视化功能。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ==================== 颜色配置 ====================

# COCO 80 类别的调色板（BGR 格式）
COLORS = [
    (56, 56, 255), (194, 77, 255), (82, 82, 255), (255, 153, 0),
    (153, 204, 0), (0, 255, 204), (128, 0, 255), (0, 128, 255),
    (64, 224, 208), (255, 0, 128), (0, 165, 255), (0, 255, 128),
    (128, 128, 255), (203, 192, 255), (128, 0, 128), (230, 25, 75),
    (245, 130, 48), (210, 245, 60), (60, 180, 75), (70, 240, 240),
    (0, 130, 200), (145, 30, 180), (255, 255, 0), (170, 255, 0),
    (0, 128, 128), (255, 216, 177), (255, 209, 153), (191, 239, 255),
    (250, 250, 55), (230, 216, 173), (128, 173, 230), (0, 0, 117),
    (240, 50, 230), (70, 70, 70), (193, 182, 255), (220, 220, 0),
    (255, 225, 25), (255, 190, 212), (170, 255, 195), (220, 190, 255),
    (176, 58, 126), (128, 128, 0), (255, 255, 179), (153, 136, 119),
    (102, 153, 153), (112, 25, 25), (25, 82, 82), (230, 0, 73),
    (93, 48, 93), (218, 179, 255), (163, 255, 0), (255, 179, 240),
    (243, 183, 178), (166, 196, 255), (208, 229, 228), (99, 106, 152),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (0, 255, 255),
    (255, 128, 0), (255, 255, 240), (0, 204, 0), (128, 255, 0),
    (255, 153, 204), (255, 153, 153), (204, 255, 153), (153, 204, 255),
    (51, 153, 255), (204, 102, 0), (128, 128, 128), (96, 164, 108),
    (108, 92, 231), (115, 17, 213), (247, 182, 210), (0, 56, 130),
    (76, 153, 0), (153, 0, 76), (102, 102, 153), (0, 128, 128),
]


def get_color(idx: int) -> Tuple[int, int, int]:
    """根据索引获取颜色"""
    return COLORS[idx % len(COLORS)]


# ==================== 检测框绘制 ====================


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    class_names: Optional[Dict[int, str]] = None,
    masks: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    conf_threshold: float = 0.25,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """在图像上绘制检测结果

    Args:
        image: BGR 图像 (H, W, 3)
        boxes: [N, 4] 边界框 (x1, y1, x2, y2) 像素坐标
        classes: [N] 类别 ID
        confidences: [N] 置信度
        class_names: {class_id: name} 类别名称映射
        masks: [N, H, W] 实例分割 mask（可选）
        keypoints: [N, K, 3] 关键点 (x, y, visible)（可选）
        conf_threshold: 低于此置信度的检测不显示
        line_thickness: 边框线宽
        font_scale: 字体大小

    Returns:
        绘制了检测结果的图像
    """
    image = image.copy()
    h, w = image.shape[:2]

    if boxes is None or len(boxes) == 0:
        return image

    for i, box in enumerate(boxes):
        conf = confidences[i] if confidences is not None else 1.0
        if conf < conf_threshold:
            continue

        cls_id = int(classes[i]) if classes is not None else 0
        color = get_color(cls_id)

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # 绘制标签
        if class_names is not None:
            label = class_names.get(cls_id, f"cls_{cls_id}")
        else:
            label = f"cls_{cls_id}"

        if confidences is not None:
            text = f"{label} {conf:.2f}"
        else:
            text = label

        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
        )
        # 标签背景
        cv2.rectangle(
            image,
            (x1, y1 - th - baseline - 3),
            (x1 + tw, y1),
            color,
            -1,
        )
        # 标签文字
        cv2.putText(
            image,
            text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_thickness,
            cv2.LINE_AA,
        )

        # 绘制 mask
        if masks is not None and i < len(masks):
            mask = masks[i]
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            colored_mask = np.zeros_like(image)
            colored_mask[mask > 0.5] = color
            image = cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0)

        # 绘制关键点
        if keypoints is not None and i < len(keypoints):
            kpts = keypoints[i]
            for kp in kpts:
                x, y, v = int(kp[0]), int(kp[1]), kp[2]
                if v > 0:
                    cv2.circle(image, (x, y), 3, color, -1)

    return image


# ==================== 训练曲线 ====================


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """绘制训练曲线

    Args:
        history: 训练历史，包含 'train/box_loss', 'train/cls_loss', 'val/box_loss' 等
        save_path: 保存路径（可选）
        figsize: 图像尺寸

    Returns:
        matplotlib Figure 对象
    """
    metrics = [k for k in history.keys() if k.startswith("train/")]
    val_metrics = [k for k in history.keys() if k.startswith("val/")]
    n = len(metrics)

    fig, axes = plt.subplots(2, max(3, (n + 1) // 2), figsize=figsize)
    axes = axes.flatten()

    for i, key in enumerate(metrics):
        ax = axes[i]
        ax.plot(history[key], label="train", linewidth=1.5)
        val_key = key.replace("train/", "val/")
        if val_key in history:
            ax.plot(history[val_key], label="val", linewidth=1.5)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # mAP 曲线
    ax = axes[n]
    for m in ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if m in history:
            ax.plot(history[m], label=m.split("/")[-1], linewidth=1.5)
    ax.set_title("mAP")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for j in range(n + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ==================== 混淆矩阵可视化 ====================


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = True,
) -> plt.Figure:
    """绘制混淆矩阵

    Args:
        matrix: [C+1, C+1] 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        figsize: 图像尺寸
        normalize: 是否归一化

    Returns:
        matplotlib Figure 对象
    """
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = matrix.astype(np.float32)
            matrix /= matrix.sum(1)[:, np.newaxis]
            matrix = np.nan_to_num(matrix)

    num_classes = matrix.shape[0] - 1
    if class_names is None:
        class_names = [f"cls_{i}" for i in range(num_classes)]
    class_names = class_names + ["background"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    # 标注数值
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.5 else "black"
            ax.text(
                j, i,
                f"{matrix[i, j]:.2f}" if normalize else f"{matrix[i, j]:.0f}",
                ha="center", va="center", fontsize=7, color=color,
            )

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 可视化自检 ===")

    # 创建测试图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (200, 200, 200)

    # 模拟检测结果
    boxes = np.array([[100, 100, 300, 400], [400, 200, 550, 350]])
    classes = np.array([0, 1])
    confidences = np.array([0.95, 0.72])
    class_names = {0: "person", 1: "car"}

    result = draw_detections(
        img, boxes, classes, confidences, class_names
    )
    print(f"检测框绘制: 输出 shape={result.shape}")

    # 模拟训练曲线
    history = {
        "train/box_loss": [0.1, 0.08, 0.06, 0.05],
        "val/box_loss": [0.12, 0.09, 0.07, 0.06],
        "metrics/mAP50(B)": [0.5, 0.6, 0.7, 0.75],
    }
    fig = plot_training_curves(history)
    plt.close(fig)
    print("训练曲线绘制成功 ✅")

    print("可视化模块就绪 ✅")
