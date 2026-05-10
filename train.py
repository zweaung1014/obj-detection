import shutil
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "detecting_adapter/data.yaml"
BASE_MODEL = "yolo11n-seg.pt"   # segmentation model — matches polygon label format
EPOCHS = 100
IMG_SIZE = 640
OUTPUT_WEIGHTS = "adapter_detector.pt"

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project="runs/segment",
    name="train",
)

# Copy best weights to project root for easy use
best_weights = Path(results.save_dir) / "weights" / "best.pt"
if best_weights.exists():
    shutil.copy(best_weights, OUTPUT_WEIGHTS)
    print(f"\nWeights saved to: {OUTPUT_WEIGHTS}")
else:
    print(f"\nWarning: best.pt not found at {best_weights}")
