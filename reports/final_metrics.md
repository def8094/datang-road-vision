# Final Model Evaluation

评估时间：2026-09-06（本地训练环境）

| Model | Algorithm | Evaluation split | Metric | Result | Target status |
|---|---|---|---|---:|---|
| `driver_emotion` | OpenCV + PyTorch CNN | RAF-DB-derived independent test | accuracy | 86.91% | 未达到 90%--95% |
| `traffic_sign` | YOLO26 | GTSRB validation | mAP50 | 97.86% | 达到 |
| `traffic_sign` | YOLO26 | GTSRB validation | mAP50-95 | 89.94% | 接近但未达到 |
| `vehicle_detection` | YOLO26 | vehicle validation | mAP50 | 92.30% | 达到 |
| `vehicle_detection` | YOLO26 | vehicle validation | mAP50-95 | 63.27% | 未达到 |
| `lane_detection` | Swin Transformer V2 + compact decoder + Lovasz/Tversky precision tuning | Complete unfiltered TuSimple validation, strict binary masks | mean IoU | 71.03% | 未达到 90%--95% |
| `lane_detection` | Swin Transformer V2 + compact decoder + Lovasz/Tversky precision tuning | Complete unfiltered TuSimple validation, strict binary masks | F1 | 83.06% | 未达到 90%--95% |
| `lane_detection` | Swin Transformer V2 + compact decoder + Lovasz/Tversky precision tuning | Complete unfiltered TuSimple validation, strict binary masks | precision / recall | 83.49% / 82.64% | 仅作辅助指标 |
| `lane_detection` | Swin Transformer V2 + compact decoder + Lovasz/Tversky precision tuning | Complete unfiltered TuSimple validation, strict binary masks | pixel accuracy | 98.56% | 仅作辅助指标 |

表情模型改用 RAF-DB 公开镜像后，独立测试准确率为 86.91%。车道线模型经过 TuSimple 训练集几何质量过滤、硬掩膜训练、Lovasz IoU 损失、Tversky 假阳性约束和冻结 Swin V2 骨干的精确化训练后，完整未过滤验证集严格 mIoU 达到 71.03%，F1 达到 83.06%。pixel accuracy 不替代 mIoU。RLMD 和 CARLA 未用于最终车道线模型的成绩汇报，因为它们的标注密度和场景分布与 TuSimple 不一致。

上述结果是当前数据划分和训练配置下的实测结果。90%--95% 是验收目标，不能在训练前保证；车道线结果为完整验证集结果，不等同于独立测试结果。新默认 checkpoint 位于 `training/lane_detection/weights/best.pt`，旧候选和混合数据模型保留在同一权重目录。
