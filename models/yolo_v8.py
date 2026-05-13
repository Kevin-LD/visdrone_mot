from ultralytics import YOLO
import os

def get_model(model_name='yolov8n.pt', task='detect', weights_dir='weights'):
    """
    模型工厂函数：加载 YOLOv8
    """
    # 1. 自动创建权重存放目录
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"创建权重目录: {weights_dir}")

    # 2. 拼接完整路径
    # 如果用户传入的是一个完整路径，则直接使用；如果是文件名，则放入 weights_dir
    if os.path.isabs(model_name) or os.sep in model_name:
        model_path = model_name
    else:
        model_path = os.path.join(weights_dir, model_name)

    try:
        # 加载模型。如果 model_path 不存在，YOLO 会自动下载到该路径
        model = YOLO(model_path, task=task)
        print(f"成功加载模型: {model_path}，任务类型: {task}")
        return model
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return None
