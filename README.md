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

This section can be extended with results from different devices.

Example:

| GPU         | Backend  | Precision | Batch | Latency (ms) | FPS |
| ----------- | -------- | --------- | ----- | ------------ | --- |
| RTX 3050 Ti | TensorRT | FP16      | 1     | 2.84         | 352 |
| RTX 2080 Ti | TensorRT | FP16      | 1     | ...          | ... |

---

## Tested hardware

* NVIDIA RTX 2080 Ti
* NVIDIA RTX 3050 Ti (laptop)
* Jetson Orin (planned)

---

## Notes

* Small batch sizes do not fully utilise the GPU
* FP16 generally improves performance on modern NVIDIA GPUs
* ONNX provides moderate improvements over PyTorch
* TensorRT delivers the highest performance due to hardware-specific optimisation

---

