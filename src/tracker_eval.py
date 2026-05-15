import cv2
import argparse
import os
import time
import numpy as np
from ultralytics import YOLO

def ccw(A, B, C):
    """
    使用叉积判断点 C 是否在线段 AB 的逆时针方向
    (B.x-A.x)*(C.y-A.y) - (B.y-A.y)*(C.x-A.x)
    """
    val = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
    if val > 0: return 1   # 逆时针
    if val < 0: return -1  # 顺时针
    return 0               # 共线

def intersect(p1, p2, p3, p4):
    """
    判断线段 (p1,p2) 与线段 (p3,p4) 是否相交
    """
    # 跨立实验：两条线段相互跨立则相交
    return (((ccw(p1, p2, p3) * ccw(p1, p2, p4)) < 0) and 
            ((ccw(p3, p4, p1) * ccw(p3, p4, p2)) < 0))

def run_tracking(args):
    # 1. 加载模型
    print(f"--- 正在加载模型: {args.model} ---")
    model = YOLO(args.model)

    # 2. 打开视频文件
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {args.input}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 设置视频写入器
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 计数器初始化
    # 基于 1920*1080 中心坐标系，调整 y 偏置为正值以定位在左下方
    # 点1: (-397, 201) -> x=960-397=563, y=540+201=741
    # 点2: (-470, 327) -> x=960-470=490, y=540+327=867
    line_pt1 = (int(width * 0.2932), int(height * 0.6861))
    line_pt2 = (int(width * 0.2552), int(height * 0.8028))
    
    counter = 0
    # 存储格式 {track_id: (last_cx, last_cy)}，用于记录上一帧的位置
    track_history = {}  

    print(f"--- 开始处理视频: {width}x{height}, FPS: {fps} ---")
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. 执行跟踪
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

        # 5. 处理越线逻辑 (只有开启 --count 时运行)
        if args.count and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                # 取底边中点作为判定点
                cx = int((box[0] + box[2]) / 2)
                cy = int(box[3]) 
                current_pos = (cx, cy)

                if track_id in track_history:
                    prev_pos = track_history[track_id]
                    
                    # 判断物体上一帧到这一帧的轨迹(线段)是否与计数线段相交
                    if intersect(prev_pos, current_pos, line_pt1, line_pt2):
                        counter += 1
                        # 可以在此打印或记录，防止同一物体在后续帧因抖动重复触发
                        # 但由于是线段相交，通常物体穿过后就不会再与该线段相交
                
                # 更新位置历史
                track_history[track_id] = current_pos

        # 6. 可视化
        annotated_frame = results[0].plot(
            line_width=2,
            font_size=0.6,
            conf=False,
            labels=True
        )

        # 绘制计数线和结果
        if args.count:
            # 选用青色 (BGR: 255, 255, 0)，粗度设为 4
            cv2.line(annotated_frame, line_pt1, line_pt2, (255, 255, 0), 4)
            # 在线段起点标注一下 L1/L2 方便调试
            cv2.putText(annotated_frame, "Count Line", (line_pt2[0]-20, line_pt2[1]+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 显示计数结果
            cv2.putText(annotated_frame, f"Count: {counter}", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"进度: {frame_count}/{total_frames} | 计数: {counter} | FPS: {frame_count/elapsed:.2f}")

    cap.release()
    out.release()
    print(f"--- 处理完成！最终总数: {counter} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 视频跟踪与越线计数脚本")
    
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--input', type=str, required=True, help='输入路径')
    parser.add_argument('--output', type=str, default='output/result.mp4', help='输出路径')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--count', action='store_true', help='是否开启越线计数功能')

    args = parser.parse_args()
    run_tracking(args)
