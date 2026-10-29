from pathlib import Path
import runpy
import shutil
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--data", str(root / "datasets/processed/traffic_sign_gtsrb/data.yaml"), "--model", "yolo26n.pt", "--project", str(root / "training/traffic_sign/runs"), "--name", "yolo26_traffic_sign", *sys.argv[1:]]
runpy.run_path(str(root / "tools/train_yolo.py"), run_name="__main__")

run_root = root / "training/traffic_sign/runs"
checkpoints = sorted(run_root.glob("yolo26_traffic_sign*/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
if checkpoints:
    destination = root / "training/traffic_sign/weights/best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoints[0], destination)
