# Luckfox Python YOLO bindings

This is a library for efficient YOLO inference on Luckfox boards. Why we can't use RKNN Toolkit lite? Because it compiled for ARM64 bit arch and RV1103/RV1106 are 32 bit (armv7l, arhmhf) cpu. And there is no source code for manual compilation.

Current status is `WIP`, for now you can use only file-based inference:

```python
yolo_api.init(MODEL_PATH)

result = yolo_api.inference(IMAGE_PATH) # file path
```