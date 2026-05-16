import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

# 1. 配置参数 overrides
args = dict(model="yolov8s.pt", data="coco8.yaml", epochs=1, imgsz=32)

# 2. 直接建立检测训练器
trainer = DetectionTrainer(overrides=args)

# 3. 显式调用设置，这步会完成模型的加载以及优化器（smart_optimizer）的构建
trainer.setup_model()
trainer.optimizer = trainer.build_optimizer(model=trainer.model)

optimizer = trainer.optimizer
model_internal = trainer.model

# 4. 建立映射表并打印
param_to_name = {id(p): name for name, p in model_internal.named_parameters()}

for i, pg in enumerate(optimizer.param_groups):
    print(f"\n⚡ [pg{i}] LR={pg.get('initial_lr', pg['lr'])}")
    for p in pg['params']:
        p_id = id(p)
        if p_id in param_to_name:
            print(f"   └── {param_to_name[p_id]}")
