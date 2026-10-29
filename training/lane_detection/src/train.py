from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = [
    "--data", str(root / "datasets/processed/lane_tusimple_filtered"),
    "--val-data", str(root / "datasets/processed/lane_tusimple_full"),
    "--batch-size", "4",
    "--workers", "4",
    "--height", "288",
    "--width", "512",
    "--pretrained",
    "--freeze-backbone-epochs", "999",
    "--hard-mask",
    "--mask-dilation", "1",
    "--positive-weight", "1.0",
    "--tversky-weight", "1.5",
    "--false-positive-weight", "0.85",
    "--lovasz-weight", "8.0",
    "--time-hours", "6",
    "--output", str(root / "training/lane_detection/weights/best.pt"),
    *sys.argv[1:],
]
runpy.run_path(str(root / "tools/train_lane_swin.py"), run_name="__main__")
