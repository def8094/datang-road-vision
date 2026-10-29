# Lane Detection

Swin Transformer V2 encoder with a segmentation decoder, trained on a quality-filtered TuSimple binary lane-line dataset. The filter removes samples with abnormal foreground ratio, vertical coverage, top-of-lane position, or connected-component count; the complete unfiltered TuSimple validation split remains the evaluation set. The selected checkpoint is `weights/best.pt`; the older mixed-data checkpoint is retained as `weights/best_mixed.pt`.

The selected model uses hard binary masks, no training-time mask dilation, positive-class weight `1.0`, Tversky false-positive weight `0.85`, and normalized Lovasz hinge weight `8.0`. The complete unfiltered TuSimple validation result is `71.03%` strict mask mIoU, `83.06%` F1, `83.49%` precision, and `82.64%` recall at threshold `0.20`. This is the best verified checkpoint in the current project, but it does not meet the requested `90%--95%` mIoU target.
