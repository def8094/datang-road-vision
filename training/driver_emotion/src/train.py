from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = [
    "--data", str(root / "datasets/processed/driver_emotion_rafdb"),
    "--val-split", "val",
    "--arch", "resnet18",
    "--image-size", "224",
    "--channels", "3",
    "--pretrained",
    "--batch-size", "64",
    "--output", str(root / "training/driver_emotion/weights/best.pt"),
    *sys.argv[1:],
]
runpy.run_path(str(root / "tools/train_emotion_cnn.py"), run_name="__main__")
