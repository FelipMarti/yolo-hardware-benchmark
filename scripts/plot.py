import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Load & preprocess
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)

    df["backend"] = df["backend"].replace({
        "pt": "PyTorch",
        "onnx": "ONNX",
        "engine": "TensorRT"
    })

    df["precision"] = df["precision"].str.replace("_gpu", "")
    df["precision"] = df["precision"].str.upper()

    df["latency_per_image"] = df["latency_ms"] / df["batch"]

    df["throughput_fps"] = df["fps"]  # (images/sec)

    gpu_name_raw = df["gpu_name"].iloc[0]
    gpu_name = gpu_name_raw.replace("_", " ")
    gpu_name_file = gpu_name_raw  

    return df, gpu_name, gpu_name_file


# -----------------------------
# Styling
# -----------------------------
def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Plot 1: Grouped Bar
# -----------------------------
def plot_grouped_bar(df, gpu_name, gpu_name_file, outdir):
    g = sns.catplot(
        data=df,
        kind="bar",
        x="batch",
        y="throughput_fps",
        hue="backend",
        col="precision",
        height=5,
        aspect=1
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Batch Size", "Throughput (images/sec)")
    g.fig.suptitle(
        f"Throughput (FPS) by Backend, Batch, and Precision\n{gpu_name}",
        y=1.05
    )

    plt.savefig(os.path.join(outdir, f"{gpu_name_file}_grouped_fps.png"))
    plt.close()


# -----------------------------
# Plot 2: Latency per Image
# -----------------------------
def plot_latency_per_image(df, gpu_name, gpu_name_file, outdir):
    g = sns.relplot(
        data=df,
        kind="line",
        x="batch",
        y="latency_per_image",
        hue="backend",
        col="precision",
        marker="o",
        height=5,
        aspect=1
    )

    g.set_axis_labels("Batch Size", "Latency per Image (ms)")
    g.fig.suptitle(
        f"Latency per Image (Efficiency)\n{gpu_name}",
        y=1.05
    )

    plt.savefig(os.path.join(outdir, f"{gpu_name_file}_latency_per_image.png"))
    plt.close()


# -----------------------------
# Plot 3: VRAM scaling
# -----------------------------
def plot_vram(df, gpu_name, gpu_name_file, outdir):
    g = sns.relplot(
        data=df,
        kind="line",
        x="batch",
        y="vram_mb",
        hue="backend",
        col="precision",
        marker="o",
        height=5,
        aspect=1
    )

    g.set_axis_labels("Batch Size", "VRAM (MB)")
    g.fig.suptitle(
        f"Memory Usage Scaling\n{gpu_name}",
        y=1.05
    )

    plt.savefig(os.path.join(outdir, f"{gpu_name_file}_vram_scaling.png"))
    plt.close()


# -----------------------------
# Plot 4: Throughput vs Latency
# -----------------------------
def plot_tradeoff(df, gpu_name, gpu_name_file, outdir):
    plt.figure()

    sns.scatterplot(
        data=df,
        x="latency_ms",
        y="throughput_fps",
        hue="backend",
        style="precision",
        size="batch",
        sizes=(50, 200)
    )

    plt.title(f"Throughput vs Latency Trade-off\n{gpu_name}")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Throughput (images/sec)")
    plt.grid()

    plt.savefig(os.path.join(outdir, f"{gpu_name_file}_tradeoff.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.outdir is None:
        outdir = repo_root / "results" / "plots"
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    setup_style()
    df, gpu_name, gpu_name_file = load_data(args.csv)


    plot_grouped_bar(df, gpu_name, gpu_name_file, outdir)
    plot_latency_per_image(df, gpu_name, gpu_name_file, outdir)
    plot_vram(df, gpu_name, gpu_name_file, outdir)
    plot_tradeoff(df, gpu_name, gpu_name_file, outdir)

    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
