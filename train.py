import argparse
from datetime import datetime
from models.yolo_v8 import get_model
import wandb
import ultralytics

def main(args):
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    full_run_name = f"{args.name}_{timestamp}"

    # 1. 初始化 WandB
    wandb.init(
        project=args.project.split('/')[-1],
        name=full_run_name, 
        config=args
    )

    print(f"--- 使用 Ultralytics v{ultralytics.__version__} 启动训练 ---")
    print(f"--- 本次实验名称: {full_run_name} ---")
    
    # 2. 获取模型
    model = get_model(args.model)
    if model is None:
        print("未能获取模型")
        wandb.finish()
        return

    # 3. 启动训练
    model.train(
        # 基础参数
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=full_run_name,
        
        # 优化器与学习率
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,     # 如果连续这么多轮 mAP 不上升则停止
        
        # 策略与增强
        multi_scale=args.multi_scale, # 开启多尺度训练
        amp=args.amp,                 # 混合精度训练
        mosaic=args.mosaic,           # Mosaic 数据增强概率
        
        # 其他固定参数
        save=True,
        exist_ok=True
    )

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 VisDrone 调优训练脚本")
    
    # --- 数据与配置参数 ---
    parser.add_argument('--data', type=str, default='configs/visdrone_data.yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='yolov8n.pt, yolov8s.pt, etc.')
    
    # --- 核心训练超参数 ---
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=640, help='建议针对 VisDrone 尝试 640 或 960')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--patience', type=int, default=50, help='早停轮数')
    
    # --- 优化器相关 ---
    parser.add_argument('--optimizer', type=str, default='auto', choices=['SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'])
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    
    # --- 训练策略 (Flags) ---
    parser.add_argument('--multi_scale', action='store_true', help='启用多尺度训练，提升尺度鲁棒性')
    parser.add_argument('--no_amp', action='store_false', dest='amp', help='禁用混合精度训练')
    parser.set_defaults(amp=True)
    
    # --- 数据增强 ---
    parser.add_argument('--mosaic', type=float, default=1.0, help='Mosaic 增强概率 (0.0 到 1.0)')
    
    # --- 硬件与监控 ---
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--project', type=str, default='weights')
    parser.add_argument('--name', type=str, default='visdrone_yolov8')

    args = parser.parse_args()

    main(args)
