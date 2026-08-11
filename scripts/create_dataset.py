"""生成本地测试数据集（coco128 无法下载时的替代方案）"""
import cv2, os, yaml, random, numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\Admin\Desktop\yolo v8\datasets\custom_test")
for d in [BASE / "images/train", BASE / "images/val",
          BASE / "labels/train", BASE / "labels/val"]:
    d.mkdir(parents=True, exist_ok=True)

CLASSES = ["person", "car", "dog", "cat", "bottle"]
COLORS = {
    "person": (255, 0, 0), "car": (0, 255, 0), "dog": (0, 0, 255),
    "cat": (255, 255, 0), "bottle": (255, 0, 255),
}

# 生成训练图片
for split, n in [("train", 40), ("val", 10)]:
    imgs_dir = BASE / f"images/{split}"
    lbls_dir = BASE / f"labels/{split}"
    for i in range(n):
        img = np.random.randint(60, 200, (640, 640, 3), dtype=np.uint8)
        labels = []
        # 每张图 2-5 个目标
        for _ in range(random.randint(2, 5)):
            cls_id = random.randint(0, len(CLASSES) - 1)
            w = random.randint(30, 150)
            h = random.randint(30, 150)
            x = random.randint(w // 2, 640 - w // 2)
            y = random.randint(h // 2, 640 - h // 2)
            color = COLORS[CLASSES[cls_id]]
            cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
            cv2.putText(img, CLASSES[cls_id], (x - w // 2, y - h // 2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            labels.append(f"{cls_id} {x/640:.6f} {y/640:.6f} {w/640:.6f} {h/640:.6f}")

        cv2.imwrite(str(imgs_dir / f"{i:04d}.jpg"), img)
        with open(lbls_dir / f"{i:04d}.txt", "w") as f:
            f.write("\n".join(labels))

# 写 YAML
yaml_path = BASE / "custom_test.yaml"
config = {
    "path": str(BASE.absolute()),
    "train": "images/train",
    "val": "images/val",
    "nc": len(CLASSES),
    "names": {i: n for i, n in enumerate(CLASSES)},
}
with open(yaml_path, "w") as f:
    yaml.dump(config, f, allow_unicode=True)

print(f"Dataset created: {BASE}")
print(f"  Train: {n} images, Val: 10 images")
print(f"  Classes: {CLASSES}")
print(f"  YAML: {yaml_path}")
print("Done!")
