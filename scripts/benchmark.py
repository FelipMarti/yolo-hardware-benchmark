import time
import argparse
import numpy as np
import torch
import csv
import os
import cv2

from ultralytics import YOLO
import onnxruntime as ort


# -----------------------------
# Utils
# -----------------------------
def get_system_info():
    info = {}

    # CUDA / GPU info
    info["cuda_available"] = torch.cuda.is_available()
    info["num_gpus"] = torch.cuda.device_count()

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "id": i,
                "name": props.name,
                "total_vram_mb": props.total_memory // 1024**2,
            })

    info["gpus"] = gpus

    return info


def load_image(imgsz):
    if os.path.exists("data/sample.jpg"):
        img = cv2.imread("data/sample.jpg")
        img = cv2.resize(img, (imgsz, imgsz))
    else:
        img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    return img


def load_batch(imgsz, batch):
    return [load_image(imgsz) for _ in range(batch)]


def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


# -----------------------------
# Benchmark functions
# -----------------------------
def benchmark_pt(model, imgs, runs=50, half=False):
    # Warmup
    for _ in range(10):
        model(imgs, verbose=False, half=half)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(runs):
        model(imgs, verbose=False, half=half)

    torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / runs * 1000
    return latency


def benchmark_onnx(session, imgs, runs=50):
    input_name = session.get_inputs()[0].name

    # Preprocess batch
    batch = []
    for img in imgs:
        x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch.append(x)

    batch = np.stack(batch)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: batch})

    start = time.time()

    for _ in range(runs):
        session.run(None, {input_name: batch})

    end = time.time()

    latency = (end - start) / runs * 1000
    return latency


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    results = []
    base = args.model  # model base name (no .pt)

    system_info = get_system_info()

    gpu_name = (
        system_info["gpus"][0]["name"]
        if system_info["gpus"]
        else "cpu"
    )
    gpu_name = gpu_name.replace(" ", "_")
    safe_gpu_name = gpu_name.replace(" ", "_").replace("/", "_")
    
    print("\n=== System Info ===")
    print(f"CUDA available: {system_info['cuda_available']}")
    print(f"Number of GPUs: {system_info['num_gpus']}")
    
    for gpu in system_info["gpus"]:
        print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['total_vram_mb']} MB)")

    for batch in [1, 4, 8]:
        print(f"\nBatch size: {batch}")
        imgs = load_batch(args.imgsz, batch)

        # -----------------
        # PyTorch FP32
        # -----------------
        pt_path = f"models/{base}.pt"
        if os.path.exists(pt_path):
            model = YOLO(pt_path, task="detect")
            lat = benchmark_pt(model, imgs, half=False)
            vram = get_vram()
            results.append((gpu_name, "pt", "fp32", batch, lat, 1000 / lat, vram))

            print(f"PT FP32: {lat:.2f} ms | {1000/lat:.2f} FPS")
        else:
            print("PT model not found")

        # -----------------
        # PyTorch FP16
        # -----------------
        if os.path.exists(pt_path):
            model = YOLO(pt_path, task="detect")
            lat = benchmark_pt(model, imgs, half=True)
            vram = get_vram()
            results.append((gpu_name, "pt", "fp16", batch, lat, 1000 / lat, vram))

            print(f"PT FP16: {lat:.2f} ms | {1000/lat:.2f} FPS")

        # -----------------
        # ONNX FP32
        # -----------------
        onnx_fp32_path = f"models/{base}_b{batch}_fp32.onnx"
        if os.path.exists(onnx_fp32_path):
            session = ort.InferenceSession(
                onnx_fp32_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            providers = session.get_providers()
            device = "GPU" if "CUDAExecutionProvider" in providers else "CPU"

            lat = benchmark_onnx(session, imgs)
            vram = get_vram()
            results.append((gpu_name,"onnx", f"fp32_{device.lower()}", batch, lat, 1000 / lat, vram))

            print(f"ONNX FP32 ({device}): {lat:.2f} ms | {1000/lat:.2f} FPS")
        else:
            print(f"ONNX model not found for batch={batch}")

        # -----------------
        # ONNX FP16
        # -----------------
        onnx_fp16_path = f"models/{base}_b{batch}_fp16.onnx"
        if os.path.exists(onnx_fp16_path):
            session = ort.InferenceSession(
                onnx_fp16_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

            providers = session.get_providers()
            device = "GPU" if "CUDAExecutionProvider" in providers else "CPU"

            lat = benchmark_onnx(session, imgs)
            vram = get_vram()
            results.append((gpu_name,"onnx", f"fp16_{device.lower()}", batch, lat, 1000 / lat, vram))

            print(f"ONNX FP16 ({device}): {lat:.2f} ms | {1000/lat:.2f} FPS")
        else:
            print(f"ONNX model not found for batch={batch}")

        # -----------------
        # TensorRT FP32
        # -----------------
        trt_fp32_path = f"models/{base}_b{batch}_fp32.engine"
        if os.path.exists(trt_fp32_path):
            model = YOLO(trt_fp32_path, task="detect")
            lat = benchmark_pt(model, imgs, half=False)
            vram = get_vram()
            results.append((gpu_name, "engine", "fp32", batch, lat, 1000 / lat, vram))

            print(f"TRT FP32: {lat:.2f} ms | {1000/lat:.2f} FPS")
        else:
            print(f"TRT FP32 engine not found for batch={batch}")

        # -----------------
        # TensorRT FP16
        # -----------------
        trt_fp16_path = f"models/{base}_b{batch}_fp16.engine"
        if os.path.exists(trt_fp16_path):
            model = YOLO(trt_fp16_path, task="detect")
            lat = benchmark_pt(model, imgs, half=True)
            vram = get_vram()
            results.append((gpu_name, "engine", "fp16", batch, lat, 1000 / lat, vram))

            print(f"TRT FP16: {lat:.2f} ms | {1000/lat:.2f} FPS")
        else:
            print(f"TRT FP16 engine not found for batch={batch}")

    # -----------------------------
    # Save CSV
    # -----------------------------
    output_path = f"results/benchmark_{safe_gpu_name}.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["gpu_name", "backend", "precision", "batch", "latency_ms", "fps", "vram_mb"]
        )
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
