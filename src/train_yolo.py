# train_yolo.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import shutil
import zipfile
import argparse
from datetime import datetime

def eprint(*args):
    print(*args, file=sys.stderr)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_existing_path(candidates):
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.normpath(c)
    return None

def fix_roboflow_data_yaml(data_yaml_path: str, data_dir: str, out_yaml_path: str) -> str:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    def resolve_split_path(p):
        if not p:
            return p
        if os.path.isabs(p) and os.path.exists(p):
            return os.path.normpath(p)

        # 常見兩種：DATA_DIR/p、DATA_DIR/<dataset_name>/p
        cand1 = os.path.normpath(os.path.join(data_dir, p))
        cand2 = os.path.normpath(os.path.join(data_dir, os.path.basename(data_dir), p))
        found = find_existing_path([cand1, cand2])
        return found if found else os.path.normpath(cand1)

    for k in ("train", "val", "test"):
        if k in data and data[k]:
            data[k] = resolve_split_path(data[k])

    # 嚴格檢查：至少 train/val 必須存在
    missing = []
    for k in ("train", "val"):
        if (k not in data) or (not data[k]) or (not os.path.exists(data[k])):
            missing.append(k)

    if missing:
        raise FileNotFoundError(
            f"data.yaml 路徑修正後仍找不到：{missing}\n"
            f"目前解析結果：train={data.get('train')}, val={data.get('val')}\n"
            f"請確認 Roboflow 下載的資料夾內容與 data.yaml。"
        )

    with open(out_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return out_yaml_path

def zip_dir(src_dir: str, zip_path: str) -> str:
    ensure_dir(os.path.dirname(zip_path))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, src_dir)
                zf.write(full, rel)
    return zip_path

def copy_if_exists(src: str, dst: str) -> bool:
    if src and os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLO with Roboflow dataset (clean, repo-friendly).")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or path (e.g., yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda / cpu / 0 / 0,1 ... (leave empty to auto)")
    parser.add_argument("--format", default="yolov8", help="Roboflow export format (yolov8 recommended)")

    # Roboflow params
    parser.add_argument("--workspace", required=True, help="Roboflow workspace (e.g., work-zuqta)")
    parser.add_argument("--project", required=True, help="Roboflow project slug (e.g., 3d-test-oye6n)")
    parser.add_argument("--version", type=int, required=True, help="Roboflow dataset version number (e.g., 16)")

    # Output paths
    parser.add_argument("--runs_dir", default="runs", help="Ultralytics runs output directory")
    parser.add_argument("--save_root", default="artifacts", help="Where to collect/export key results")
    parser.add_argument("--name", default=None, help="Experiment name (default auto timestamp)")

    args = parser.parse_args()

    # Import here so the script can show clear error if deps missing
    try:
        from ultralytics import YOLO, settings
        from roboflow import Roboflow
    except Exception as ex:
        eprint("❌ 缺少套件。請先安裝：pip install ultralytics roboflow pyyaml")
        raise

    rf_api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not rf_api_key:
        raise EnvironmentError(
            "找不到環境變數 ROBOFLOW_API_KEY。\n"
            "請先在命令列設定：\n"
            "  Windows PowerShell:  $env:ROBOFLOW_API_KEY='你的key'\n"
            "  Windows CMD:         set ROBOFLOW_API_KEY=你的key\n"
            "  macOS/Linux:         export ROBOFLOW_API_KEY='你的key'\n"
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = args.name or f"yolo_{args.project}_v{args.version}_{stamp}"
    runs_dir = os.path.abspath(args.runs_dir)
    save_dir = os.path.abspath(os.path.join(args.save_root, exp_name))
    ensure_dir(runs_dir)
    ensure_dir(save_dir)

    print("===== Settings =====")
    print("exp_name :", exp_name)
    print("runs_dir :", runs_dir)
    print("save_dir :", save_dir)
    print("model    :", args.model)
    print("epochs   :", args.epochs)
    print("imgsz    :", args.imgsz)
    print("batch    :", args.batch)
    print("device   :", args.device)

    # Make ultralytics use our runs dir
    settings.update({"runs_dir": runs_dir})

    # 1) Download dataset from Roboflow
    print("\n⬇️ Downloading dataset from Roboflow ...")
    rf = Roboflow(api_key=rf_api_key)
    proj = rf.workspace(args.workspace).project(args.project)
    ver = proj.version(args.version)
    dataset = ver.download(args.format)  # returns object with .location
    data_dir = dataset.location
    data_yaml_org = os.path.join(data_dir, "data.yaml")
    if not os.path.exists(data_yaml_org):
        raise FileNotFoundError(f"找不到 data.yaml：{data_yaml_org}")

    print("DATA_DIR        :", data_dir)
    print("data.yaml (org) :", data_yaml_org)

    # 2) Fix yaml -> absolute paths
    data_yaml_fixed = os.path.join(data_dir, "data_abs.yaml")
    print("\n🛠 Fixing data.yaml to absolute paths ...")
    fix_roboflow_data_yaml(data_yaml_org, data_dir, data_yaml_fixed)
    print("data.yaml (fixed):", data_yaml_fixed)

    # 3) Train
    print("\n🚀 Training ...")
    model = YOLO(args.model)
    train_res = model.train(
        data=data_yaml_fixed,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=exp_name,
        project=runs_dir,
    )

    # Ultralytics usually creates: runs/detect/<exp_name>
    run_dir = os.path.join(runs_dir, "detect", exp_name)
    if not os.path.exists(run_dir):
        # fallback: sometimes it can be runs/<task>/<name> depending on config
        # We'll search under runs_dir for a folder ending with exp_name
        candidates = []
        for root, dirs, _ in os.walk(runs_dir):
            for d in dirs:
                if d == exp_name:
                    candidates.append(os.path.join(root, d))
        if candidates:
            run_dir = candidates[0]

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"找不到訓練輸出資料夾 run_dir：{run_dir}")

    print("run_dir:", run_dir)

    # 4) Val (use best.pt if exists)
    best_pt = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        print("\n✅ Found best.pt, running validation ...")
        model_best = YOLO(best_pt)
    else:
        print("\n⚠️ best.pt not found, validating with current model weights ...")
        model_best = model

    val_name = f"{exp_name}_val"
    val_res = model_best.val(
        data=data_yaml_fixed,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=val_name,
        project=runs_dir,
    )

    val_dir = os.path.join(runs_dir, "detect", val_name)
    if not os.path.exists(val_dir):
        # optional; not fatal
        val_dir = None

    # 5) Collect artifacts
    print("\n📦 Collecting artifacts ...")

    # weights
    for w in ("best.pt", "last.pt"):
        src = os.path.join(run_dir, "weights", w)
        dst = os.path.join(save_dir, f"train_{w}")
        copy_if_exists(src, dst)

    # plots/metrics (train + val)
    common_files = [
        "results.png", "results.csv",
        "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png",
        "confusion_matrix.png", "confusion_matrix_normalized.png",
        "labels.jpg", "labels_correlogram.jpg",
    ]

    for fn in common_files:
        copy_if_exists(os.path.join(run_dir, fn), os.path.join(save_dir, f"train_{fn}"))
        if val_dir:
            copy_if_exists(os.path.join(val_dir, fn), os.path.join(save_dir, f"val_{fn}"))

    # Save a small summary text
    summary_path = os.path.join(save_dir, "SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Roboflow: workspace={args.workspace}, project={args.project}, version={args.version}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}, device: {args.device}\n")
        f.write(f"Run dir: {run_dir}\n")
        if val_dir:
            f.write(f"Val dir: {val_dir}\n")
        f.write("\nVal metrics (results_dict):\n")
        try:
            f.write(str(val_res.results_dict) + "\n")
        except Exception:
            f.write("N/A\n")

    # Zip save_dir
    zip_path = os.path.join(args.save_root, f"{exp_name}.zip")
    zip_path = os.path.abspath(zip_path)
    zip_dir(save_dir, zip_path)

    print("\n🎉 Done.")
    print("Artifacts folder:", save_dir)
    print("Zip file        :", zip_path)
    print("Ultralytics run :", run_dir)

if __name__ == "__main__":
    main()
