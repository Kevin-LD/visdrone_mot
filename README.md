# Scene Object Detection and Multi-Object Tracking
## 项目简介
本项目基于 YOLOv8 与 ByteTrack 实现场景目标检测与视频多目标跟踪，包含 VisDrone 数据集微调训练、视频 MOT 推理以及越线计数功能。  
## 环境配置
实验环境：Ubuntu 22.04.5 LTS（WSL） + Python 3.12。  
使用以下命令安装项目依赖：  
```bash
pip install -r requirements.txt
```
## 数据准备
运行以下脚本下载并预处理 VisDrone2019-DET 数据集：  
```bash
python src/data_prep.py
```
## 运行方式
### 模型训练
```bash
python src/train.py --model weights/yolov8s.pt --imgsz 752 --batch 8
```
### 视频多目标跟踪
```bash
python src/tracker_eval.py --model path/to/model.pt --input path/to/video.mp4 --output output/path.mp4 --imgsz 2176
```
### 越线计数
```bash
python src/tracker_eval.py --model path/to/model.pt --input path/to/video.mp4 --output output/path.mp4 --imgsz 2176 --count
```
可通过修改 `src/tracker_eval.py` 中的 `line_pt1` 与 `line_pt2` 调整 counting line 位置。  
更多可选参数可通过以下命令查看：  
```bash
python src/train.py -h
python src/tracker_eval.py -h
```
## 模型权重
模型权重下载链接：[Google Drive Link](https://drive.google.com/file/d/1VD7jnPL8jcp6kPRprn_ptsq_WHy5IRDW/view?usp=sharing)  
## 相关仓库
- Task 1: [Pet Classification with Transfer Learning](https://github.com/Kevin-LD/pet_classification_transfer_learning)
- Task 3: [Pixel-level-Training-Of-Image-Segmentation](https://github.com/F1shermanCNN/Pixel-level-Training-Of-Image-Segmentation#pixel-level-training-of-image-segmentation)
- Report: [Computer Vision Midterm Report](https://github.com/Kevin-LD/cv_midterm_report) 
