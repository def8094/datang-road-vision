from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--model", str(root / "training/traffic_sign/runs/yolo26_traffic_sign/weights/best.pt"), *sys.argv[1:]]
runpy.run_path(str(root / "tools/infer_yolo.py"), run_name="__main__")
