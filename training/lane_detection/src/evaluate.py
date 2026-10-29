from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--data", str(root / "datasets/processed/lane_tusimple_full"), "--checkpoint", str(root / "training/lane_detection/weights/best.pt"), "--split", "val", "--mask-dilation", "1", *sys.argv[1:]]
runpy.run_path(str(root / "tools/evaluate_lane.py"), run_name="__main__")
