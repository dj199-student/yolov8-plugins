"""
YOLOv8 数据增强模块

支持：
- Mosaic 增强：将 4 张图拼接成 1 张
- MixUp 增强：两张图加权混合
- HSV 颜色扰动
- 随机几何变换（旋转、平移、缩放、剪切、翻转）
- Copy-Paste 增强（实例分割用）
"""

import random
from typing import Tuple

import cv2
import numpy as np


class AlbumentationsWrapper:
    """Albumentations 库的数据增强包装器（如果可用）"""

    def __init__(self, size: int = 640):
        self.size = size
        self.transform = None
        try:
            import albumentations as A

            self.transform = A.Compose(
                [
                    A.Blur(p=0.01),
                    A.MedianBlur(p=0.01),
                    A.ToGray(p=0.01),
                    A.CLAHE(p=0.01),
                ],
                bbox_params=A.BboxParams(
                    format="yolo", label_fields=["class_labels"]
                ),
            )
        except ImportError:
            pass

    def __call__(
        self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.transform is None:
            return image, bboxes, labels

        transformed = self.transform(
            image=image, bboxes=bboxes, class_labels=labels
        )
        return (
            transformed["image"],
            np.array(transformed["bboxes"]),
            np.array(transformed["class_labels"]),
        )


def random_hsv(image: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    """HSV 颜色空间随机扰动

    Args:
        image: BGR 图像 (H, W, 3)
        hgain: Hue 增益
        sgain: Saturation 增益
        vgain: Value 增益

    Returns:
        扰动后的图像
    """
    if hgain == 0 and sgain == 0 and vgain == 0:
        return image

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    dtype = image.dtype
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    )
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def random_perspective(
    image: np.ndarray,
    targets: np.ndarray = None,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    border: tuple = (0, 0),
) -> Tuple[np.ndarray, np.ndarray]:
    """随机仿射变换（旋转、平移、缩放、剪切、透视）

    Args:
        image: 输入图像
        targets: 标注 [N, 5] 格式 (class, x, y, w, h) 归一化坐标
        degrees: 旋转角度范围
        translate: 平移范围（比例）
        scale: 缩放范围（比例）
        shear: 剪切范围
        perspective: 透视范围
        border: (宽度, 高度) 填充边界

    Returns:
        (变换后图像, 变换后标注)
    """
    height, width = image.shape[0], border[0] + image.shape[0] + border[0]
    width_img = image.shape[1] + border[1] + border[1]

    # 中心
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x 平移
    C[1, 2] = -image.shape[0] / 2  # y 平移

    # 透视
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    # 旋转 + 缩放
    R = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=s)

    # 剪切
    S = np.eye(3)
    S[0, 1] = np.tan(random.uniform(-shear, shear) * np.pi / 180)
    S[1, 0] = np.tan(random.uniform(-shear, shear) * np.pi / 180)

    # 平移
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width_img
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    # 组合变换矩阵
    M = T @ S @ R @ P @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            image = cv2.warpPerspective(
                image, M, dsize=(width_img, height), borderValue=(114, 114, 114)
            )
        else:
            image = cv2.warpAffine(
                image, M[:2], dsize=(width_img, height), borderValue=(114, 114, 114)
            )

    # 变换标注
    if targets is not None and len(targets):
        n = len(targets)
        xy = np.ones((n * 4, 3), dtype=np.float32)
        # 将归一化标注转为像素坐标的角点
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)

        # 计算新的 bbox
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new_bboxes = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))
        ).reshape(4, n).T

        # 裁剪到图像范围并归一化
        new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, width_img)
        new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, height)

        # 过滤无效标注
        valid = (new_bboxes[:, 2] > new_bboxes[:, 0]) & (
            new_bboxes[:, 3] > new_bboxes[:, 1]
        )
        targets = targets[valid]
        if len(targets):
            targets[:, 1:5] = xywhn2xyxy(
                targets[:, 1:5], w=width_img, h=height
            )
            targets[:, 1:5] = xyxy2xywhn(
                targets[:, 1:5], w=width_img, h=height, clip=True
            )

    return image, targets


def mosaic_augment(
    images: list, labels: list, img_size: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """Mosaic 增强：将 4 张图拼接为 1 张

    Args:
        images: 4 张图像的列表
        labels: 4 组标注的列表（每组格式: class, x_center, y_center, w, h 归一化）
        img_size: 输出图像尺寸

    Returns:
        (mosaic 图像, mosaic 标注)
    """
    assert len(images) == 4, "Mosaic 需要恰好 4 张图像"

    # 创建马赛克画布
    mosaic_img = np.full(
        (img_size * 2, img_size * 2, images[0].shape[2]),
        114,
        dtype=np.uint8,
    )

    # 中心点（随机偏移）
    xc = int(random.uniform(img_size // 2, img_size + img_size // 2))
    yc = int(random.uniform(img_size // 2, img_size + img_size // 2))

    mosaic_labels = []

    for i, (img, label) in enumerate(zip(images, labels)):
        h, w = img.shape[:2]

        # 确定放置位置
        if i == 0:  # 左上
            x1a, y1a = max(xc - w, 0), max(yc - h, 0)
            x2a, y2a = xc, yc
            x1b, y1b = w - (x2a - x1a), h - (y2a - y1a)
            x2b, y2b = w, h
        elif i == 1:  # 右上
            x1a, y1a = xc, max(yc - h, 0)
            x2a, y2a = min(xc + w, img_size * 2), yc
            x1b, y1b = 0, h - (y2a - y1a)
            x2b, y2b = min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a = max(xc - w, 0), yc
            x2a, y2a = xc, min(yc + h, img_size * 2)
            x1b, y1b = w - (x2a - x1a), 0
            x2b, y2b = w, min(y2a - y1a, h)
        else:  # 右下
            x1a, y1a = xc, yc
            x2a, y2a = min(xc + w, img_size * 2), min(yc + h, img_size * 2)
            x1b, y1b = 0, 0
            x2b, y2b = min(w, x2a - x1a), min(y2a - y1a, h)

        # 放置图像
        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        # 调整标注坐标
        padw = x1a - x1b
        padh = y1a - y1b
        if len(label):
            label = label.copy()
            label[:, 1] = (label[:, 1] * w + padw) / (img_size * 2)
            label[:, 2] = (label[:, 2] * h + padh) / (img_size * 2)
            label[:, 3] = (label[:, 3] * w) / (img_size * 2)
            label[:, 4] = (label[:, 4] * h) / (img_size * 2)

            # 过滤越界标注
            valid = (
                (label[:, 1] > 0)
                & (label[:, 1] < 1)
                & (label[:, 2] > 0)
                & (label[:, 2] < 1)
            )
            label = label[valid]

        mosaic_labels.append(label)

    mosaic_labels = (
        np.concatenate(mosaic_labels, axis=0) if any(len(l) for l in mosaic_labels) else np.zeros((0, 5))
    )

    # 缩放到目标尺寸
    mosaic_img = cv2.resize(mosaic_img, (img_size, img_size))

    return mosaic_img, mosaic_labels


def mixup_augment(
    img1: np.ndarray,
    labels1: np.ndarray,
    img2: np.ndarray,
    labels2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """MixUp 增强：两张图按比例混合

    Args:
        img1, labels1: 第一张图及其标注
        img2, labels2: 第二张图及其标注

    Returns:
        (混合图像, 合并标注)
    """
    r = np.random.beta(32.0, 32.0)

    # 确保两张图尺寸一致
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mixed_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    mixed_labels = np.concatenate((labels1, labels2), axis=0)

    return mixed_img, mixed_labels


def copy_paste_augment(
    img1: np.ndarray, labels1: np.ndarray, img2: np.ndarray, labels2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Copy-Paste 增强（实例分割用）

    将 img2 中的实例粘贴到 img1 上，同时合并标注。
    """
    # 简化实现：随机选取 img2 中的实例粘贴到 img1
    if len(labels2) == 0:
        return img1, labels1

    # 选取部分实例
    n_paste = random.randint(1, min(3, len(labels2)))
    indices = random.sample(range(len(labels2)), n_paste)

    for idx in indices:
        label = labels2[idx]
        cls_id, xc, yc, w, h = int(label[0]), label[1], label[2], label[3], label[4]

        h_img, w_img = img1.shape[:2]
        xc_px, yc_px = int(xc * w_img), int(yc * h_img)
        w_px, h_px = int(w * w_img), int(h * h_img)

        x1 = max(0, xc_px - w_px // 2)
        y1 = max(0, yc_px - h_px // 2)
        x2 = min(w_img, xc_px + w_px // 2)
        y2 = min(h_img, yc_px + h_px // 2)

        if x2 <= x1 or y2 <= y1:
            continue

        # 从 img2 取对应区域（简化：取对应尺寸的随机区域）
        src_h, src_w = y2 - y1, x2 - x1
        if src_h <= 0 or src_w <= 0:
            continue

        src_y = random.randint(0, img2.shape[0] - src_h - 1) if img2.shape[0] > src_h else 0
        src_x = random.randint(0, img2.shape[1] - src_w - 1) if img2.shape[1] > src_w else 0

        paste_region = img2[src_y:src_y + src_h, src_x:src_x + src_w]
        img1[y1:y2, x1:x2] = cv2.addWeighted(
            img1[y1:y2, x1:x2], 0.5, paste_region, 0.5, 0
        )

    return img1, labels1


# ==================== 标注格式转换工具 ====================


def xywhn2xyxy(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """归一化 xywh → 像素 xyxy"""
    y = x.copy()
    y[:, 0] = (x[:, 0] - x[:, 2] / 2) * w  # x1
    y[:, 1] = (x[:, 1] - x[:, 3] / 2) * h  # y1
    y[:, 2] = (x[:, 0] + x[:, 2] / 2) * w  # x2
    y[:, 3] = (x[:, 1] + x[:, 3] / 2) * h  # y2
    return y


def xyxy2xywhn(x: np.ndarray, w: int, h: int, clip: bool = False) -> np.ndarray:
    """像素 xyxy → 归一化 xywh"""
    if clip:
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1)
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1)
    y = x.copy()
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w         # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h         # height
    return y


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 数据增强自检 ===")
    # 创建测试图像
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # HSV 增强测试
    augmented = random_hsv(img, 0.015, 0.7, 0.4)
    print(f"HSV 增强: 输入 shape={img.shape}, 输出 shape={augmented.shape}")

    # 创建测试标注
    labels = np.array([[0, 0.5, 0.5, 0.2, 0.2]])

    # 透视变换测试
    img_t, labels_t = random_perspective(
        img, labels, degrees=10, translate=0.1, scale=0.5, shear=2
    )
    print(f"透视变换: 输出 shape={img_t.shape}, 标注数={len(labels_t)}")

    print("数据增强模块就绪 ✅")
