# Smart Self-Checkout System

A smart self-checkout POS demo using **YOLO-based product recognition** and **Arduino weight sensing**.

## Structure
- `src/` Python programs (POS, server, detector, training, export)
- `data/` product and transaction data (CSV)
- `weights/` trained YOLO model (`best.pt`, `best.onnx`)
- `static/` web UI files
- `sketch_oct3a/` Arduino weight sensing code

## Requirements
- Python 3.10+
- Windows 10 / 11

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/server.py
python src/detector_client.py
python src/pos.py
```

## Model

The trained model is already included in weights/, so the demo can run without re-training
