from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "tools"))
sys.argv[1:] = ["--detector", str(root / "training/vehicle_detection/runs/yolo26_vehicle/weights/best.pt"), *sys.argv[1:]]
runpy.run_path(str(root / "tools/run_deepsort.py"), run_name="__main__")
