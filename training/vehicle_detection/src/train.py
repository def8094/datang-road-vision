from pathlib import Path
import runpy
import shutil
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--data", str(root / "datasets/processed/vehicle_detection_3class/data.yaml"), "--model", "yolo26n.pt", "--project", str(root / "training/vehicle_detection/runs"), "--name", "yolo26_vehicle", "--batch", "4", *sys.argv[1:]]
runpy.run_path(str(root / "tools/train_yolo.py"), run_name="__main__")

run_root = root / "training/vehicle_detection/runs"
checkpoints = sorted(run_root.glob("yolo26_vehicle*/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
if checkpoints:
    destination = root / "training/vehicle_detection/weights/best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoints[0], destination)
