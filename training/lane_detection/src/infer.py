from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--checkpoint", str(root / "training/lane_detection/weights/best.pt"), *sys.argv[1:]]
runpy.run_path(str(root / "tools/infer_lane.py"), run_name="__main__")
