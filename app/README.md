# PySide6 实时检测界面

## 运行环境

项目默认使用 Conda 的 `smart` 环境。交付时请使用项目根目录的
`environment.smart.yml` 在目标电脑创建一个属于目标用户自己的环境；不要求目标电脑使用原电脑的绝对路径。

当前参考环境包含：

- Python 3.13.5
- PyTorch 2.9.1 + CUDA 12.8
- OpenCV 4.12.0
- PySide6 6.10.1

在 PyCharm 中把解释器设置为目标电脑实际创建的环境，例如：

C:\\Users\\你的用户名\\.conda\\envs\\road_smart\\python.exe

运行配置的脚本路径为：

F:\\大唐道路\\app\\main.py

工作目录设置为：

F:\\大唐道路

## 启动

PowerShell：

    .\\app\\run_smart.ps1

脚本会调用当前已经激活的 Conda 环境中的 `python`，并设置
`PYTHONNOUSERSITE=1`，避免用户目录里的旧 Python 包覆盖当前环境。
也可以显式指定解释器：

    .\\app\\run_smart.ps1 -PythonExecutable "C:\\Users\\你的用户名\\.conda\\envs\\road_smart\\python.exe"

或直接运行：

    conda activate road_smart
    $env:PYTHONNOUSERSITE = "1"
    python app\\main.py

## 功能

- 上传 MP4、AVI、MOV、MKV 视频并逐帧检测。
- 车辆检测使用项目根目录的 YOLO26 COCO 权重 `yolo26n.pt`，车辆轨迹使用 DeepSORT。
- `training/vehicle_detection/weights/best.pt` 保留为历史训练产物；该权重对应的数据集类别名未得到可靠确认，且在用户道路视频上实测无车辆输出，因此不作为实时界面默认权重。
- 交通标识使用 YOLO26 锚框检测。
- 交通标识推理会过滤低置信度、接近整幅画面的 crop-like 框，避免 GTSRB 裁剪数据造成道路场景大框误报。
- 对明显的红色圆形标识增加 OpenCV 形状候选框兜底；该候选框表示已发现标识形状，不冒充 GTSRB 类别名称。
- 驾驶员表情使用 OpenCV 人脸框和 PyTorch ResNet18 分类。
- 车道线使用 Swin Transformer V2 分割，并在画面中显示绿色区域与青色锚点。
- 默认优先使用 CUDA；设备下拉框可切换 CPU。
- 四个模型按需加载，避免启动时一次性占用全部显存。

当前车道线界面默认优先使用：

training\\lane_detection\\weights\\candidate_highres_precision.pt

该候选模型在完整 TuSimple 验证集上的严格 mIoU 为 78.23%。如果该候选不存在，界面回退到 training\\lane_detection\\weights\\best.pt。
