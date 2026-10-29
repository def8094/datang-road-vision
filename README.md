# 大唐道路 · 自动驾驶视觉检测系统（datang-road-vision）

> 4 人团队桌面端行车视频检测软件 —— PySide6 实时界面，模块化集成 **车辆检测跟踪（YOLO26 + DeepSORT）**、**交通标识检测（YOLO26）**、**驾驶员表情识别（ResNet18）**、**车道线分割（Swin Transformer V2）** 四大功能，支持 CUDA/CPU 切换与按功能勾选降载。

上传视频后，界面实时叠加：车辆框 + DeepSORT 轨迹锚点、交通标识框、人脸表情标签、车道线分割掩膜与可行驶区域锚点。

## ✨ 功能特性

| 模块 | 说明 |
|---|---|
| 🚗 车辆检测与跟踪 | YOLO26 检测（COCO 车辆五类）+ NMS 过滤 + DeepSORT 跟踪（卡尔曼滤波 + 匈牙利算法 + 轨迹管理） |
| 🚸 交通标识检测 | YOLO26 检测 GTSRB 43 类，ROI 级红圈标识二次校验过滤误检 |
| 😀 驾驶员表情识别 | Haar 人脸检测 + ImageNet 预训练 ResNet18，七类表情分类 |
| 🛣 车道线分割 | Swin Transformer V2 编码器 + 分割解码器，掩膜阈值化提取可行驶区域，水平扫描生成车道锚点 |
| 🖥 桌面界面 | 模型按需懒加载、推理独立 QThread 工作线程防 UI 卡顿、CUDA/CPU 回退、按功能勾选降载 |

## 🛠 技术栈

- **语言**：Python 3.13
- **界面**：PySide6（QThread 线程隔离、信号槽渲染）
- **检测跟踪**：Ultralytics YOLO26、deep-sort-realtime
- **分割分类**：PyTorch、torchvision（Swin V2 / ResNet18）
- **视频处理**：OpenCV（MP4 / AVI / MOV / MKV）

## 📊 模型指标

以下为独立验证集评估结果（非训练集分数）：

| 模型 | Checkpoint | 评估方式 | 结果 |
|---|---|---|---:|
| 驾驶员表情 | `training/driver_emotion/weights/best.pt` | RAF-DB 派生测试集准确率 | **86.91%** |
| 交通标识 | `training/traffic_sign/runs/yolo26_traffic_sign-2/weights/best.pt` | YOLO26 mAP50 | **97.86%** |
| 车辆检测 + DeepSORT | `training/vehicle_detection/runs/yolo26_vehicle/weights/best.pt` | YOLO26 mAP50 | **92.30%** |
| 车道线分割 | `training/lane_detection/weights/best.pt` | 完整未过滤 TuSimple 验证集 mIoU | **71.03%** |

## 🚀 快速开始

### 1. 环境准备

- Windows 10/11
- Python 3.12/3.13（建议 venv 或 Conda）
- NVIDIA GPU（可选）：需 cu128 版 PyTorch（如 `torch==2.9.1+cu128`）；无 GPU 自动回退 CPU

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install pyside6
```

> GPU 用户注意：PyPI Windows 版 torch 是 CPU 版，需从 PyTorch 官方索引安装 cu128 构建：
> ```bash
> .venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
> ```

### 3. 启动界面

```bash
.venv\Scripts\python app\main.py
```

Windows 下也可直接双击仓库根目录的 `启动检测界面.bat`。

### 4. 使用

1. 点击「打开视频」，选择本地视频文件
2. 勾选需要启用的功能模块（默认全开）
3. 点开始检测；状态栏显示逐帧推理延迟

## 📁 目录结构

```
datang-road-vision/
├── app/                    # PySide6 桌面界面 + 统一推理管理器
│   ├── main.py             # 窗口 / 视频线程 / 渲染
│   └── inference.py        # 四模型懒加载、逐帧标注、DeepSORT
├── tools/                  # 数据预处理 / 训练 / 评估 / 联调脚本
│   ├── prepare_datasets.py # 数据集解压与报告
│   ├── train_yolo.py       # YOLO26 训练入口
│   ├── train_emotion_cnn.py
│   ├── train_lane_swin.py  # Swin V2 车道线训练
│   ├── evaluate_*.py       # 独立验证集评估
│   └── run_deepsort.py     # 车辆跟踪推理 CLI
├── training/               # 各模型训练配置与产物说明（权重不入库）
├── configs/
├── reports/                # 数据清单与指标报告
└── docs/                   # 启动指南 / 算法说明
```

## 🔧 关于权重文件

模型权重（`.pt`）因体积原因不入库，其中车道线 Swin V2 单文件 139MB 超过 GitHub 100MB 限制。获取方式二选一：

1. **自行训练**：参照 `docs/` 与 `tools/` 下的脚本，数据集准备命令见下；
2. **联系作者**获取打包权重（yolo26n.pt 可由 Ultralytics 自动下载）。

数据准备与训练入口：

```bash
python tools/prepare_datasets.py --report-only
python tools/prepare_datasets.py --extract
python training/driver_emotion/src/train.py     # 或 tools/ 下对应脚本
python training/traffic_sign/src/train.py
python training/vehicle_detection/src/train.py
python training/lane_detection/src/train.py
```

## ⚙️ 工程要点

- **数据契约**：四模块共享统一的逐帧 stats 字典（`vehicle / vehicle_tracks / traffic_sign / emotion / lane / latency_ms`），界面渲染与推理解耦
- **YOLO26 硬约束**：代码校验 checkpoint 的 `yaml_file` 必须含 `yolo26`，拒绝 yolov5/yolo11 等旧版权重
- **显存控制**：懒加载 + 按功能勾选，四模型并行推理下界面流畅
- **已知坑**：OpenCV 在 Windows 上无法读取非 ASCII 路径的 Haar 级联 XML，`inference.py` 已做 ASCII 临时目录兼容

## 📌 局限与后续

- 表情识别 86.91% 未达 90% 目标，后续可引入 FERPlus 加权与均衡采样
- 车道线 mIoU 71.03%（完整未过滤 TuSimple），可尝试更大分辨率输入与 UPerNet 解码头
- 旧车辆检测权重类别契约未确认，界面默认使用 YOLO26 COCO 权重，待补齐标注后重训
