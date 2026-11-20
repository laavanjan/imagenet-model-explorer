"""Latency / size benchmark for every model in the zoo.

Uses randomly initialised weights (``pretrained=False``): inference cost depends only on the
architecture, so nothing needs to be downloaded. Accuracy numbers come from torchvision's
published metadata for the pretrained weights, not from this script.

    python benchmark.py                # prints a markdown table
    python benchmark.py --write        # also writes docs/benchmarks.md
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from classifier import MODEL_SPECS, ImageClassifier


def bench(key: str, runs: int, warmup: int = 3) -> dict:
    clf = ImageClassifier(key, device="cpu", pretrained=False)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8))
    for _ in range(warmup):
        clf.predict(img, top_k=5)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        clf.predict(img, top_k=5)
        times.append((time.perf_counter() - t0) * 1000)
    clf.grad_cam(img, 0)  # warm-up: the first backward pass allocates autograd state
    cam_times = []
    for _ in range(max(3, runs // 4)):
        t0 = time.perf_counter()
        clf.grad_cam(img, 0)
        cam_times.append((time.perf_counter() - t0) * 1000)
    cam_ms = statistics.median(cam_times)
    s = clf.spec
    return {
        "name": s.name,
        "params": s.params_m,
        "acc1": s.acc1,
        "input": clf.input_size,
        "median": statistics.median(times),
        "p95": float(np.percentile(times, 95)),
        "cam": cam_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--write", action="store_true", help="write docs/benchmarks.md")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    rows = [bench(k, args.runs) for k in MODEL_SPECS]

    header = (
        f"CPU inference, batch size 1, {args.runs} runs, top-5. "
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, {platform.processor() or platform.machine()}."
    )
    lines = [
        "| Model | Params (M) | ImageNet top-1 (%) | Input | Median latency (ms) | p95 (ms) | Grad-CAM (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['params']:.1f} | {r['acc1']:.1f} | {r['input']}² | {r['median']:.0f} | {r['p95']:.0f} | {r['cam']:.0f} |"
        )
    table = "\n".join(lines)
    print(header + "\n\n" + table)

    if args.write:
        out = Path(__file__).parent / "docs" / "benchmarks.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(f"# Benchmarks\n\n{header}\n\n{table}\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
