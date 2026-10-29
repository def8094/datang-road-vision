from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
run_root = root / "training/traffic_sign/runs"
checkpoints = sorted(run_root.glob("yolo26_traffic_sign*/weights/best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
checkpoint = checkpoints[0] if checkpoints else run_root / "yolo26_traffic_sign/weights/best.pt"
sys.argv[1:] = ["--data", str(root / "datasets/processed/traffic_sign_gtsrb/data.yaml"), "--model", str(checkpoint), *sys.argv[1:]]
runpy.run_path(str(root / "tools/evaluate_yolo.py"), run_name="__main__")
