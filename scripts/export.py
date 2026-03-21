# scripts/export.py

import argparse
from ultralytics import YOLO
import os
import torch, sys

print("PYTHON:", sys.executable)
print("CUDA:", torch.cuda.is_available())


BATCH_SIZES = [1, 4, 8] 


def export_onnx(model_path, imgsz, batch):
    model = YOLO(model_path)

    base = os.path.basename(model_path).replace(".pt", "")
    out_path = f"models/{base}_b{batch}.onnx"

    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        batch=batch,
        dynamic=False
    )

    default_path = model_path.replace(".pt", ".onnx")
    if os.path.exists(default_path):
        os.rename(default_path, out_path)

    print(f"Exported ONNX (batch={batch}) -> {out_path}")


def export_engine(model_path, imgsz, batch):
    model = YOLO(model_path)

    base = os.path.basename(model_path).replace(".pt", "")

    # -------------------------
    # FP32 export
    # -------------------------
    out_path_fp32 = f"models/{base}_b{batch}_fp32.engine"

    model.export(
        format="engine",
        imgsz=imgsz,
        batch=batch,
        half=False,
        device=0
    )

    default_path = model_path.replace(".pt", ".engine")
    if os.path.exists(default_path):
        os.rename(default_path, out_path_fp32)

    print(f"Exported TRT FP32 (batch={batch}) -> {out_path_fp32}")

    # -------------------------
    # FP16 export
    # -------------------------
    out_path_fp16 = f"models/{base}_b{batch}_fp16.engine"

    model.export(
        format="engine",
        imgsz=imgsz,
        batch=batch,
        half=True,
        device=0
    )

    if os.path.exists(default_path):
        os.rename(default_path, out_path_fp16)

    print(f"Exported TRT FP16 (batch={batch}) -> {out_path_fp16}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # Ensure model exists locally
    model = YOLO(args.model)
    pt_path = f"models/{args.model}"
    model.save(pt_path)

    for batch in BATCH_SIZES:
        print(f"\nExporting batch={batch}")

        export_onnx(pt_path, args.imgsz, batch)
        export_engine(pt_path, args.imgsz, batch)


if __name__ == "__main__":
    main()
