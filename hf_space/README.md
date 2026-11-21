---
title: Image Classifier
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Image Classifier

Upload a photo and a pretrained **ConvNeXt-Tiny** (82.5 % top-1 on ImageNet-1k) names what it sees.
A Grad-CAM heatmap shows which parts of the picture drove the answer, and the app warns you when the
model is only guessing (the object is probably not one of the 1,000 ImageNet categories).
