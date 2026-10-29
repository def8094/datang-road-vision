# Algorithm Traceability

本文档记录项目实现与课程资料的对应关系。课程资料中的示例代码和版本可能较旧，用户明确指定的版本约束优先。

| Model | Source document evidence | Implemented algorithm | Version decision |
|---|---|---|---|
| `driver_emotion` | `驾驶员表情识别.pdf`: FER2013, OpenCV, compact CNN | OpenCV grayscale/equalization + PyTorch CNN | Keras example translated to PyTorch as required by the project stack |
| `traffic_sign` | `自动驾驶-交通标识识别与可视化.pdf`: traffic-sign detection and YOLO route | YOLO26 detection, GTSRB CSV ROI converted to YOLO boxes | User explicitly prohibited YOLOv5/YOLO11 and fixed YOLO26; the code rejects other YOLO versions |
| `vehicle_detection` | `自动驾驶 - 副本.docx`: vehicle detection and DeepSORT | YOLO26 detection + `deep-sort-realtime` at inference | DeepSORT is a tracker, so it has no separate supervised checkpoint |
| `lane_detection` | `自动驾驶-车道线检测.pdf`: Swin Transformer detection/segmentation route | Pretrained Swin Transformer V2 encoder + PyTorch lane-mask decoder | Compact implementation selected for the supplied paired masks and 8 GiB GPU |

## Non-training data

`datasets/raw/vehicle_images_unlabeled` contains KITTI `image_2` images but no `label_2` annotations. It is therefore retained as unlabelled material and excluded from supervised training.

## Acceptance rule

The 90%--95% target is a target, not a value that can be guaranteed before evaluation. Final acceptance must use the held-out validation metrics recorded by the evaluation scripts; no metric is fabricated when the dataset or model does not reach the target.
