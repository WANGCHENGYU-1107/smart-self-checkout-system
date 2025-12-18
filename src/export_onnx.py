# export_onnx.py
# -*- coding: utf-8 -*-

import os
import argparse
import shutil

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Export YOLO weights to ONNX (clean, standalone).")
    parser.add_argument("--weights", default="weights/best.pt", help="Path to .pt weights")
    parser.add_argument("--outdir", default="exports", help="Output directory")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic ONNX")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--device", default=None, help="cuda / cpu / 0 ... (leave empty to auto)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        raise RuntimeError("缺少套件 ultralytics。請先安裝：pip install ultralytics")

    weights = os.path.abspath(args.weights)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"找不到權重檔：{weights}")

    ensure_dir(args.outdir)
    outdir = os.path.abspath(args.outdir)

    print("weights :", weights)
    print("outdir  :", outdir)
    print("dynamic :", args.dynamic)
    print("imgsz   :", args.imgsz)
    print("device  :", args.device)

    model = YOLO(weights)

    # Ultralytics export will return the exported path
    exported_path = model.export(format="onnx", dynamic=args.dynamic, imgsz=args.imgsz, device=args.device)
    exported_path = os.path.abspath(exported_path)

    # Put a copy into outdir with a predictable name
    base = os.path.splitext(os.path.basename(weights))[0]
    final_path = os.path.join(outdir, f"{base}.onnx")
    shutil.copy2(exported_path, final_path)

    print("\n✅ ONNX exported:")
    print(" - Ultralytics path:", exported_path)
    print(" - Copied to       :", final_path)
    print("\n(你可以用 Netron 打開 .onnx 視覺化)")

if __name__ == "__main__":
    main()
