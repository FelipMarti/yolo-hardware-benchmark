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
    
    # FP32 export
    out_path_fp32 = f"models/{base}_b{batch}_fp32.onnx"

    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        batch=batch,
        dynamic=False,
        half=False
    )

    default_path = model_path.replace(".pt", ".onnx")
    if os.path.exists(default_path):
        os.rename(default_path, out_path_fp32)

    print(f"Exported ONNX FP32 (batch={batch}) -> {out_path_fp32}")


    # FP16 export
    out_path_fp16 = f"models/{base}_b{batch}_fp16.onnx"
    
    model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        batch=batch,
        dynamic=False,
        half=True  
    )
    
    # rename default output if needed
    default_path = model_path.replace(".pt", ".onnx")
    if os.path.exists(default_path):
        os.rename(default_path, out_path_fp16)
    
    print(f"Exported ONNX FP16 (batch={batch}) -> {out_path_fp16}")


def export_engine(model_path, imgsz, batch):
    model = YOLO(model_path)

    base = os.path.basename(model_path).replace(".pt", "")

    # FP32 export
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

    # FP16 export
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

    pt_path = f"models/{args.model}"
    model = YOLO(pt_path)

    for batch in BATCH_SIZES:
        print(f"\nExporting batch={batch}")

        export_onnx(pt_path, args.imgsz, batch)
        export_engine(pt_path, args.imgsz, batch)


if __name__ == "__main__":
    main()
