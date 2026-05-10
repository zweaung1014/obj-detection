# Object Detection with YOLOv11

Real-time object detection using a webcam, combining a general-purpose YOLO model with a custom-trained adapter/device detector.

## What it does

- **Live webcam feed** — captures frames from your webcam in real time
- **General object detection** — uses `yolo11n.pt` to detect everyday objects (people, cups, chairs, etc.)
- **Custom device detection** — uses a fine-tuned `adapter_detector.pt` model to detect a specific device/adapter class on top of the general detections
- Both sets of detections are overlaid on the same frame and displayed in a window

## Files

| File | Description |
|---|---|
| `object_detector.py` | Run real-time detection using both models |
| `train.py` | Fine-tune `yolo11n-seg.pt` on the custom adapter dataset |
| `detecting_adapter/data.yaml` | Dataset config for the custom training |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python
```

You will also need the model weights (not included in the repo):
- `yolo11n.pt` — download from [Ultralytics](https://github.com/ultralytics/assets/releases)
- `yolo11n-seg.pt` — same source, segmentation variant
- `adapter_detector.pt` — produced by running `train.py`

## Training the custom model

```bash
python train.py
```

Trains for 100 epochs on the adapter dataset and saves the best weights as `adapter_detector.pt`.

## Running detection

```bash
python object_detector.py
```

Press `q` to quit.
