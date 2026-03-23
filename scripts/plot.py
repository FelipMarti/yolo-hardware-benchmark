import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

    return df


# -----------------------------
# Styling
# -----------------------------
def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)


# -----------------------------
# Plot 1: Grouped Bar (BEST)
# -----------------------------
def plot_grouped_bar(df, outdir):
    g = sns.catplot(
        data=df,
        kind="bar",
        x="batch",
        y="fps",
        hue="backend",
        col="precision",
        height=5,
        aspect=1
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("Batch Size", "FPS")
    g.fig.suptitle("Throughput (FPS) by Backend, Batch, and Precision", y=1.05)

    plt.savefig(os.path.join(outdir, "grouped_fps.png"))
    plt.close()


# -----------------------------
# Plot 2: Latency per Image
# -----------------------------
def plot_latency_per_image(df, outdir):
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
    g.fig.suptitle("Efficiency: Latency per Image", y=1.05)

    plt.savefig(os.path.join(outdir, "latency_per_image.png"))
    plt.close()


# -----------------------------
# Plot 3: VRAM scaling
# -----------------------------
def plot_vram(df, outdir):
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
    g.fig.suptitle("Memory Usage Scaling", y=1.05)

    plt.savefig(os.path.join(outdir, "vram_scaling.png"))
    plt.close()


# -----------------------------
# Plot 4: Throughput vs Latency
# -----------------------------
def plot_tradeoff(df, outdir):
    plt.figure()

    sns.scatterplot(
        data=df,
        x="latency_ms",
        y="fps",
        hue="backend",
        style="precision",
        size="batch",
        sizes=(50, 200)
    )

    plt.title("Throughput vs Latency Tradeoff")
    plt.xlabel("Latency (ms)")
    plt.ylabel("FPS")
    plt.grid()

    plt.savefig(os.path.join(outdir, "tradeoff.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    setup_style()
    df = load_data(args.csv)

    plot_grouped_bar(df, args.outdir)
    plot_latency_per_image(df, args.outdir)
    plot_vram(df, args.outdir)
    plot_tradeoff(df, args.outdir)

    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
