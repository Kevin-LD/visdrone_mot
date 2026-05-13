import os
import cv2
import random
import numpy as np
from pathlib import Path

# 定义 VisDrone 类别名称 (对应 data_prep.py 中的 0-9)
CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

# 为每个类别分配一个随机颜色用于绘图
COLORS = [tuple(random.randint(0, 255) for _ in range(3)) for _ in CLASS_NAMES]

def plot_yolo_sample(img_path, label_path, save_path=None):
    """
    读取一张图片和其 YOLO 格式标签，并绘制 BBox
    """
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    h, w, _ = img.shape

    # 读取标签
    if not label_path.exists():
        print(f"Warning: Label file not found for {img_path.name}")
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        cls_id = int(parts[0])
        # 反归一化坐标: [x_center, y_center, width, height] -> [x1, y1, x2, y2]
        x_c, y_c, bw, bh = map(float, parts[1:])
        
        x1 = int((x_c - bw / 2) * w)
        y1 = int((y_c - bh / 2) * h)
        x2 = int((x_c + bw / 2) * w)
        y2 = int((y_c + bh / 2) * h)

        # 绘制矩形框
        color = COLORS[cls_id] if cls_id < len(COLORS) else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签
        label_text = f"{CLASS_NAMES[cls_id]}"
        cv2.putText(img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 显示结果
    if save_path:
        cv2.imwrite(str(save_path), img)
        print(f"Visualization saved to: {save_path}")
    else:
        cv2.imshow("Data Verification", img)
        cv2.waitKey(0)

def main():
    # 配置路径
    YOLO_DATA_DIR = Path("./data/yolo_visdrone")
    IMG_DIR = YOLO_DATA_DIR / "images" / "train"
    LBL_DIR = YOLO_DATA_DIR / "labels" / "train"
    OUTPUT_DIR = Path("./figures")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMG_DIR.exists():
        print(f"Error: Directory {IMG_DIR} does not exist. Please run data_prep.py first.")
        return

    # 随机选取 5 张图片进行测试
    all_images = list(IMG_DIR.glob("*.jpg"))
    if not all_images:
        print("No images found in the target directory.")
        return
        
    sample_images = random.sample(all_images, min(5, len(all_images)))

    print(f"Starting verification for {len(sample_images)} samples...")
    for img_p in sample_images:
        lbl_p = LBL_DIR / (img_p.stem + ".txt")
        save_p = OUTPUT_DIR / f"data_visualization_{img_p.name}"
        plot_yolo_sample(img_p, lbl_p, save_p)

    cv2.destroyAllWindows()
    print("Verification complete. Please check the 'test/output_samples' folder.")

if __name__ == "__main__":
    main()
