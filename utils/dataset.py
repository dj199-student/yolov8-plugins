"""
YOLOv8 数据集构建工具

提供：
- 数据集路径检测和验证
- YOLO 格式数据集构建
- 自定义数据集注册
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml


# ==================== YOLO 数据集格式 ====================

# YOLO 数据集标准目录结构：
# dataset/
# ├── images/
# │   ├── train/
# │   └── val/
# ├── labels/
# │   ├── train/
# │   └── val/
# └── dataset.yaml


def create_dataset_yaml(
    dataset_path: Union[str, Path],
    class_names: List[str],
    save_path: Optional[Union[str, Path]] = None,
) -> Dict:
    """创建 YOLO 格式的数据集配置文件

    Args:
        dataset_path: 数据集根目录
        class_names: 类别名称列表
        save_path: 保存 YAML 文件的路径（可选）

    Returns:
        数据集配置字典
    """
    dataset_path = Path(dataset_path)

    config = {
        "path": str(dataset_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dataset_path / "images" / "test").exists() else None,
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    # 移除空值
    config = {k: v for k, v in config.items() if v is not None}

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"数据集配置已保存: {save_path}")

    return config


def validate_dataset(dataset_path: Union[str, Path]) -> Tuple[bool, str]:
    """验证数据集格式是否正确

    Args:
        dataset_path: 数据集根目录或 YAML 文件路径

    Returns:
        (是否有效, 错误消息)
    """
    dataset_path = Path(dataset_path)

    # 如果是 YAML 文件
    if dataset_path.suffix in (".yaml", ".yml"):
        if not dataset_path.exists():
            return False, f"YAML 文件不存在: {dataset_path}"

        with open(dataset_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_keys = ["train", "val", "nc", "names"]
        for key in required_keys:
            if key not in config:
                return False, f"缺少必要字段: '{key}'"

        nc = config["nc"]
        names = config["names"]
        if len(names) != nc:
            return False, f"类别数不一致: nc={nc}, len(names)={len(names)}"

        # 检查图像路径
        base_path = Path(config.get("path", ""))
        for split in ["train", "val"]:
            if split in config:
                img_dir = base_path / config[split]
                if not img_dir.exists():
                    # 尝试 dataset_path 的父目录
                    img_dir = dataset_path.parent / config[split]
                if not img_dir.exists():
                    return False, f"{split} 图像目录不存在: {img_dir}"

        return True, "数据集配置有效 ✅"

    return False, f"请提供 YAML 配置文件路径，收到: {dataset_path}"


# ==================== 标注格式转换 ====================


def voc_to_yolo(
    voc_xml_dir: Union[str, Path],
    output_dir: Union[str, Path],
    class_names: List[str],
) -> int:
    """将 VOC XML 格式标注转为 YOLO txt 格式

    Args:
        voc_xml_dir: VOC XML 标注目录
        output_dir: 输出目录
        class_names: 类别名称列表

    Returns:
        转换的文件数
    """
    import xml.etree.ElementTree as ET

    voc_xml_dir = Path(voc_xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    count = 0

    for xml_file in voc_xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            continue
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in class_to_idx:
                continue
            cls_id = class_to_idx[name]

            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)

            # 转为 YOLO 格式 (class, x_center, y_center, width, height) 归一化
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        txt_file = output_dir / f"{xml_file.stem}.txt"
        with open(txt_file, "w") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"转换完成: {count} 个文件, 类别数={len(class_names)}")
    return count


def coco_to_yolo(
    coco_json: Union[str, Path],
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> int:
    """将 COCO JSON 格式标注转为 YOLO txt 格式

    Args:
        coco_json: COCO JSON 标注文件路径
        image_dir: 图像目录
        output_dir: 输出目录

    Returns:
        转换的文件数
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # 建立映射
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_to_idx = {cat_id: i for i, cat_id in enumerate(categories.keys())}
    images = {img["id"]: img for img in coco["images"]}

    # 按 image_id 组织标注
    annotations_by_image: Dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    count = 0
    for img_id, img_info in images.items():
        w, h = img_info["width"], img_info["height"]
        anns = annotations_by_image.get(img_id, [])

        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_to_idx:
                continue

            cls_id = cat_to_idx[cat_id]
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        img_name = Path(img_info["file_name"])
        txt_file = output_dir / f"{img_name.stem}.txt"
        with open(txt_file, "w") as f:
            f.write("\n".join(lines))
        count += 1

    print(f"COCO→YOLO 转换完成: {count} 个文件, {len(categories)} 个类别")
    return count


# ==================== 数据集统计 ====================


def dataset_statistics(
    label_dir: Union[str, Path], class_names: List[str]
) -> Dict:
    """统计数据集信息

    Args:
        label_dir: 标注文件目录（YOLO txt 格式）
        class_names: 类别名称列表

    Returns:
        包含类别分布、bbox 尺寸等统计信息的字典
    """
    label_dir = Path(label_dir)
    class_counts = np.zeros(len(class_names), dtype=int)
    widths, heights = [], []

    for txt_file in label_dir.glob("*.txt"):
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    w, h = float(parts[3]), float(parts[4])
                    if cls_id < len(class_names):
                        class_counts[cls_id] += 1
                    widths.append(w)
                    heights.append(h)

    widths = np.array(widths) if widths else np.zeros(0)
    heights = np.array(heights) if heights else np.zeros(0)

    return {
        "total_instances": int(class_counts.sum()),
        "class_distribution": {
            name: int(count) for name, count in zip(class_names, class_counts)
        },
        "bbox_width": {
            "mean": float(widths.mean()) if len(widths) else 0,
            "std": float(widths.std()) if len(widths) else 0,
            "min": float(widths.min()) if len(widths) else 0,
            "max": float(widths.max()) if len(widths) else 0,
        },
        "bbox_height": {
            "mean": float(heights.mean()) if len(heights) else 0,
            "std": float(heights.std()) if len(heights) else 0,
        },
    }


# ==================== 自检 ====================

if __name__ == "__main__":
    print("=== 数据集工具自检 ===")

    # 创建测试数据集 YAML
    config = create_dataset_yaml(
        "/tmp/test_dataset",
        ["person", "car", "dog", "cat"],
    )
    print(f"数据集配置: nc={config['nc']}, names={config['names']}")

    # 验证
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(config, f)
        tmp_path = f.name

    valid, msg = validate_dataset(tmp_path)
    print(f"验证结果: {msg}")
    os.unlink(tmp_path)

    print("数据集工具就绪 ✅")
