import cv2
import argparse
import os
import time
from ultralytics import YOLO

def run_tracking(args):
    # 1. 加载模型 (自动识别检测任务)
    print(f"--- 正在加载模型: {args.model} ---")
    model = YOLO(args.model)

    # 2. 打开视频文件
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {args.input}")
        return

    # 获取视频元数据
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 设置视频写入器
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"--- 开始处理视频: {width}x{height}, FPS: {fps} ---")
    print(f"--- 结果将保存至: {args.output} ---")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. 执行跟踪
        # persist=True 保证跟踪器在帧之间保持状态
        # tracker 参数可选 'botsort.yaml' 或 'bytetrack.yaml'
        results = model.track(
            source=frame,
            persist=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            tracker=args.tracker,
            device=args.device,
            verbose=False
        )

        # 5. 可视化结果
        # 调整 line_width 为 1，font_size 适当减小 (如 0.5 到 0.8)
        annotated_frame = results[0].plot(
            line_width=2,       # 减小框的线条宽度
            font_size=0.6,     # 减小字体大小
            conf=False,         # 如果不需要看置信度，设为 False 可以让标签更短
            labels=True        # 保持显示类别和ID
        )

        # 写入帧
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            curr_fps = frame_count / elapsed
            print(f"进度: {frame_count}/{total_frames} 帧 | 当前平均 FPS: {curr_fps:.2f}")

    # 6. 释放资源
    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"--- 处理完成！总耗时: {total_time:.2f}s, 平均 FPS: {frame_count/total_time:.2f} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 视频跟踪评估脚本")
    
    # 路径参数
    parser.add_argument('--model', type=str, required=True, help='训练好的 best.pt 模型路径')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, default='output/result.mp4', help='保存视频路径')
    
    # 跟踪超参数
    parser.add_argument('--imgsz', type=int, default=640, help='推理分辨率')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU 阈值')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', choices=['botsort.yaml', 'bytetrack.yaml'])
    
    # 硬件参数
    parser.add_argument('--device', type=str, default='0', help='cpu 或 0, 1...')

    args = parser.parse_args()
    run_tracking(args)
