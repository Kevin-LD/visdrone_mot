import argparse
from datetime import datetime

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.yolo_v8 import get_model

import ultralytics
import os

def main(args):
    # 1. 判断是否为 Resume 模式
    # 如果指定了 resume_path，则认为进入断点续训模式
    is_resume = args.resume_path is not None

    if is_resume:
        if not os.path.exists(args.resume_path):
            print(f"错误: checkpoint 不存在: {args.resume_path}")
            return

        # 从 checkpoint 路径推断实验名称
        # 例如: runs/detect/exp/weights/last.pt -> exp
        run_dir = os.path.dirname(os.path.dirname(args.resume_path))
        full_run_name = os.path.basename(run_dir)

        print(f"--- 恢复训练模式 ---")
        print(f"--- Checkpoint: {args.resume_path} ---")
        print(f"--- Run 名称: {full_run_name} ---")

        # 直接加载 checkpoint
        model = get_model(args.resume_path)

        if model is None:
            print("错误: 未能成功加载 checkpoint")
            return

    # 2. 正常训练模式
    else:
        # 生成带有时间戳的实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        full_run_name = f"{args.name}_{timestamp}"

        print(f"--- 启动训练 ---")
        print(f"--- Ultralytics 版本: {ultralytics.__version__} ---")
        print(f"--- WandB 项目: {args.project} ---")
        print(f"--- 实验 Run 名称: {full_run_name} ---")
        
        # 获取模型
        model = get_model(args.model)
        if model is None:
            print("错误: 未能成功加载模型，请检查权重路径。")
            return

    # 3. 启动训练
    # YOLOv8 内部的回调函数会自动处理 wandb.init()
    model.train(
        # 基础数据参数
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,

        # 路径与监控
        project=args.project,
        name=full_run_name,

        # 优化器与学习率
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,

        # 训练策略
        multi_scale=args.multi_scale,
        amp=args.amp,
        mosaic=args.mosaic,

        # Resume 状态
        resume=is_resume,

        # 落地参数
        save=True,
        exist_ok=True,
        plots=True  # 确保生成训练曲线图并上传到 WandB
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 VisDrone 自动托管训练脚本")
    
    # 数据与配置参数
    parser.add_argument('--data', type=str, default='configs/dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='weights/yolov8n.pt', help='预训练权重路径')

    # Resume 参数：合并为单一参数
    parser.add_argument('--resume_path', type=str, default=None, 
                        help='checkpoint 路径（如需断点续训请指定此参数，例如 runs/detect/xxx/weights/last.pt）')

    # 核心训练超参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--patience', type=int, default=50, help='早停轮数')

    # 优化器相关
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'])
    parser.add_argument('--lr0', type=float, default=0.01)

    # 训练策略
    parser.add_argument('--multi_scale', action='store_true', help='多尺度训练')
    parser.add_argument('--no_amp', action='store_false', dest='amp', help='禁用混合精度')
    parser.set_defaults(amp=True)

    # 数据增强
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic 增强概率')

    # 硬件与监控
    parser.add_argument('--device', type=str, default='0', help='GPU 设备 ID')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')

    # WandB 项目配置
    parser.add_argument('--project', type=str, default='visdrone_yolov8', help='WandB 项目名称')
    parser.add_argument('--name', type=str, default='task2_run', help='实验名称前缀')
    
    args = parser.parse_args()
    main(args)
