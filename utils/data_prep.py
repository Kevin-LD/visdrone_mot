import os
import requests
import zipfile
from pathlib import Path
import cv2
from tqdm import tqdm

# 数据集下载链接
VISDRONE_URLS = {
    'VisDrone2019-DET-train': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip',
    'VisDrone2019-DET-val': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip'
}

def download_file(url, save_path):
    """
    下载文件
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(save_path)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def check_and_download_visdrone(raw_data_path):
    """
    检测目标路径是否包含原始数据，不包含则下载并解压
    """
    raw_path = Path(raw_data_path)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    for folder_name, url in VISDRONE_URLS.items():
        folder_path = raw_path / folder_name
        
        # 检查文件夹是否存在且不为空
        if folder_path.exists() and any(folder_path.iterdir()):
            print(f"Data already exists in: {folder_path}")
            continue
        
        # 开始下载
        zip_path = raw_path / f"{folder_name}.zip"
        print(f"Data not found. Starting download for {folder_name}...")
        try:
            download_file(url, zip_path)
            
            # 解压文件
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_path)
            
            # 删除压缩包以节省空间
            os.remove(zip_path)
            print(f"Successfully downloaded and extracted {folder_name}")
            
        except Exception as e:
            print(f"Error downloading {folder_name}: {e}")
            print("Please manually download the dataset and place it in the data/raw_visdrone directory.")

def convert_visdrone_to_yolo(img_size, box):
    """
    坐标转换逻辑：像素坐标 [L, T, w, h] -> YOLO 归一化中心坐标 [x_c, y_c, w, h]
    """
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def process_visdrone_dataset(src_dir, dest_dir, mode='train'):
    """
    处理 VisDrone 数据集，将其转换为 YOLO 格式
    """
    img_src = Path(src_dir) / 'images'
    anno_src = Path(src_dir) / 'annotations'
    
    img_dest = Path(dest_dir) / 'images' / mode
    label_dest = Path(dest_dir) / 'labels' / mode
    
    img_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)
    
    anno_files = list(anno_src.glob('*.txt'))
    
    print(f"Converting {mode} split...")
    for anno_file in tqdm(anno_files):
        img_file = img_src / (anno_file.stem + '.jpg')
        if not img_file.exists():
            continue
            
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w, _ = img.shape
        
        # 建立硬链接节省空间
        new_img_path = img_dest / (anno_file.stem + '.jpg')
        if not new_img_path.exists():
            os.link(str(img_file), str(new_img_path))
            
        yolo_annos = []
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8: continue
                category = int(parts[5])
                if category in [0, 11]: continue # 过滤无关类别
                
                yolo_cat = category - 1 # 类别标签从 0 开始
                box = [float(p) for p in parts[:4]]
                yolo_box = convert_visdrone_to_yolo((w, h), box)
                yolo_annos.append(f"{yolo_cat} {' '.join([f'{x:.6f}' for x in yolo_box])}")
        
        with open(label_dest / (anno_file.stem + '.txt'), 'w') as f:
            f.write('\n'.join(yolo_annos))

if __name__ == "__main__":
    RAW_DATA_PATH = './data/raw_visdrone'
    YOLO_DATA_PATH = './data/yolo_visdrone'
    
    # 1. 检查并自动下载
    check_and_download_visdrone(RAW_DATA_PATH)
    
    # 2. 遍历处理各数据集分支
    splits = {
        'train': 'VisDrone2019-DET-train',
        'val': 'VisDrone2019-DET-val'
    }
    
    for mode, folder_name in splits.items():
        src_split = os.path.join(RAW_DATA_PATH, folder_name)
        if os.path.exists(src_split):
            process_visdrone_dataset(src_split, YOLO_DATA_PATH, mode)
