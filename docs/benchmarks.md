# Benchmarks

CPU inference, batch size 1, 20 runs, top-5. torch 2.15.0+cpu, 4 threads, Intel64 Family 6 Model 142 Stepping 10, GenuineIntel.

| Model | Params (M) | ImageNet top-1 (%) | Input | Median latency (ms) | p95 (ms) | Grad-CAM (ms) |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 25.6 | 80.9 | 224² | 125 | 136 | 374 |
| ResNet-101 | 44.5 | 81.9 | 224² | 207 | 217 | 722 |
| ConvNeXt-Tiny | 28.6 | 82.5 | 224² | 98 | 108 | 351 |
| ConvNeXt-Small | 50.2 | 83.6 | 224² | 175 | 187 | 639 |
| EfficientNet-V2-S | 21.5 | 84.2 | 384² | 231 | 243 | 919 |
