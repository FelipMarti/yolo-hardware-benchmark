# YOLO Hardware Benchmark

A small playground for testing how different optimisation techniques affect YOLO inference performance across hardware.

## Overview

This repository benchmarks YOLO models using several inference backends and precision modes to understand how optimisation choices impact latency, throughput, and memory usage.

The focus is on practical, reproducible comparisons across:

* PyTorch (baseline inference)
* ONNX Runtime (graph-level optimisation)
* TensorRT (hardware-level optimisation)

### Optimisation techniques explored

* FP32 vs FP16 precision
* ONNX export and runtime execution
* TensorRT engine generation
* Batch size scaling

The goal is not maximum performance at all costs, but to observe how each optimisation step changes behaviour on different devices.

---

## Requirements

This project is designed to run on NVIDIA GPUs with CUDA support.

### System requirements

* Python **3.12**
* NVIDIA GPU
* CUDA (compatible with your PyTorch / ONNX Runtime / TensorRT versions)

### Notes

* GPU acceleration is required for meaningful benchmarking results.
* ONNX Runtime and TensorRT will fall back to CPU if CUDA is not available, which will significantly impact performance.
* The scripts have been tested on:

  * Desktop GPUs (e.g. RTX 2080 Ti)
  * Laptop GPUs (e.g. RTX 3050 Ti)
  * Jetson devices (planned)

### Verify your setup

You can quickly check that CUDA is available:

```bash id="z8a3nl"
python -c "import torch; print(torch.cuda.is_available())"
```

This should return:

```id="v1m7ru"
True
```

If it returns `False`, your CUDA setup is not correctly configured.

---

## Setup

```bash
git clone https://github.com/felipmarti/yolo-hardware-benchmark.git
cd yolo-hardware-benchmark

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Prepare model

Place a YOLO model inside the `models/` directory:

```bash
models/yolo11n.pt
```

---

## Export models

Export ONNX and TensorRT engines for multiple batch sizes:

```bash
python scripts/export.py --model yolo11n.pt --imgsz 640
```

This generates one model per batch size and precision:

```
models/
  yolo11n_b1_fp32.onnx
  yolo11n_b1_fp16.onnx
  yolo11n_b1_fp32.engine
  yolo11n_b1_fp16.engine

  yolo11n_b4_fp32.onnx
  yolo11n_b4_fp16.onnx
  yolo11n_b4_fp32.engine
  yolo11n_b4_fp16.engine

  yolo11n_b8_fp32.onnx
  yolo11n_b8_fp16.onnx
  yolo11n_b8_fp32.engine
  yolo11n_b8_fp16.engine
```

Each file corresponds to:

* A fixed batch size (`b1`, `b4`, `b8`)
* A precision (`fp32`, `fp16`)
* A backend format (`onnx`, `engine`)

---

## Run benchmark

```bash
python scripts/benchmark.py --model yolo11n --imgsz 640
```

Example output:

```
PT FP32: 6.32 ms | 158 FPS
ONNX FP16 (GPU): 5.86 ms | 170 FPS
TRT FP16: 2.84 ms | 352 FPS
```

Results are saved to:

```
results/benchmark_<GPU_NAME>.csv
```

---

## What is measured

Each configuration is evaluated using:

* Latency (milliseconds)
* Throughput (frames per second)
* GPU memory usage (VRAM)

### Test dimensions

* Batch sizes: 1, 4, 8
* Precision:

  * FP32
  * FP16

### Backends

| Backend  | Description                                   |
| -------- | --------------------------------------------- |
| PyTorch  | Baseline inference without graph optimisation |
| ONNX     | Graph optimisation via ONNX Runtime           |
| TensorRT | Hardware-specific optimisation and execution  |

---

## Results

###  RTX 3050 Ti (Laptop)

| Batch | Backend  | Precision  | Latency (ms) | FPS        | VRAM (MB) |
| ----- | -------- | ---------- | ------------ | ---------- | --------- |
| 1     | PyTorch  | FP32       | 6.66         | 150.05     | 42.10     |
| 1     | PyTorch  | FP16       | 6.00         | 166.68     | 46.94     |
| 1     | ONNX     | FP32       | 7.23         | 138.26     | 46.94     |
| 1     | ONNX     | FP16       | 6.20         | 161.25     | 46.94     |
| 1     | TensorRT | FP32       | 4.88         | 204.71     | 51.39     |
| 1     | TensorRT | FP16       | **2.84**     | **352.70** | 51.39     |
| 4     | PyTorch  | FP32       | 22.27        | 44.90      | 42.10     |
| 4     | PyTorch  | FP16       | 15.35        | 65.15      | 46.94     |
| 4     | ONNX     | FP32       | 28.05        | 35.65      | 46.94     |
| 4     | ONNX     | FP16       | 20.78        | 48.12      | 46.94     |
| 4     | TensorRT | FP32       | 16.70        | 59.87      | 76.47     |
| 4     | TensorRT | FP16       | **9.66**     | **103.55** | 76.47     |
| 8     | PyTorch  | FP32       | 44.61        | 22.42      | 56.24     |
| 8     | PyTorch  | FP16       | 29.37        | 34.05      | 61.08     |
| 8     | ONNX     | FP32       | 57.00        | 17.55      | 61.08     |
| 8     | ONNX     | FP16       | 43.04        | 23.23      | 61.08     |
| 8     | TensorRT | FP32       | 32.73        | 30.56      | 120.51    |
| 8     | TensorRT | FP16       | **18.41**    | **54.32**  | 120.51    |


###  RTX 2080 Ti (Desktop)

| Batch | Backend  | Precision  | Latency (ms) | FPS        | VRAM (MB) |
| ----- | -------- | ---------- | ------------ | ---------- | --------- |
| 1     | PyTorch  | FP32       | 6.08         | 164.58     | 42.10     |
| 1     | PyTorch  | FP16       | 6.65         | 150.34     | 46.94     |
| 1     | ONNX     | FP32       | 5.41         | 184.92     | 46.94     |
| 1     | ONNX     | FP16       | 4.85         | 206.30     | 46.94     |
| 1     | TensorRT | FP32       | 3.34         | 299.44     | 51.39     |
| 1     | TensorRT | FP16       | **2.54**     | **392.94** | 51.39     |
| 4     | PyTorch  | FP32       | 11.24        | 88.93      | 42.10     |
| 4     | PyTorch  | FP16       | 10.66        | 93.78      | 46.94     |
| 4     | ONNX     | FP32       | 17.10        | 58.46      | 46.94     |
| 4     | ONNX     | FP16       | 15.00        | 66.68      | 46.94     |
| 4     | TensorRT | FP32       | 9.59         | 104.26     | 76.47     |
| 4     | TensorRT | FP16       | **7.07**     | **141.42** | 76.47     |
| 8     | PyTorch  | FP32       | 20.97        | 47.68      | 56.24     |
| 8     | PyTorch  | FP16       | 16.11        | 62.07      | 61.08     |
| 8     | ONNX     | FP32       | 34.62        | 28.88      | 61.08     |
| 8     | ONNX     | FP16       | 29.91        | 33.43      | 61.08     |
| 8     | TensorRT | FP32       | 17.91        | 55.82      | 120.51    |
| 8     | TensorRT | FP16       | **12.87**    | **77.69**  | 120.51    |


---

## Notes

* Small batch sizes do not fully utilise the GPU
* FP16 generally improves performance on modern NVIDIA GPUs
* ONNX provides moderate improvements over PyTorch
* TensorRT delivers the highest performance due to hardware-specific optimisation

---

