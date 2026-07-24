# detect_code — YOLO11n NCNN 球检测

把训练好的球检测模型（YOLO11n，单类 `ball`，NCNN 格式，三规格在 `models/best_ncnn_{320,416,640}/`）
部署到树莓派 5 上跑。推理走原生 `ncnn`（轻量、不拖入 torch，反映真实部署帧率）。

## 依赖

```bash
/home/hao/vision_env/bin/pip install ncnn   # 一次性
```

## 脚本

| 脚本 | 作用 |
|---|---|
| `yolo_ncnn.py` | 推理封装：letterbox → 前向 → 解码 + NMS → 框还原回原图。被下面两个复用，不单独跑 |
| `detect_live.py` | 实时检测预览（复用固定相机参数 + 默认去畸变）。需显示器/桌面会话 |
| `benchmark.py` | 三规格性能基准（用录像帧回放，**不需**摄像头/显示器，可纯 SSH 跑） |

```bash
# 性能基准
/home/hao/vision_env/bin/python3 code/detect_code/benchmark.py --iters 200

# 实时检测（默认 320；运行时 1/2/3 切规格、u 切去畸变、q 退出）
/home/hao/vision_env/bin/python3 code/detect_code/detect_live.py --size 320
```

## 要点

- **默认去畸变**：`detect_live.py` 有 `camera_calib.npz` 就默认喂去畸变画面（与 v1.2 /
  record_dataset 一致，部署与训练几何一致）；固定曝光/增益/白平衡复用
  `code/ready_code/camera_common.py`（唯一事实来源）。
- **输出布局**：ncnn 导出的 `out0` 是 `[5, N]` = `[cx,cy,w,h,置信度]`，框坐标已是输入尺度像素、
  已过 Sigmoid（DFL/anchor 解码烘焙进图），后处理不需手写 anchor 解码。
- **坑**：`ncnn.Mat(numpy)` 只包装不拷贝缓冲区，预处理数组出作用域会段错误，`yolo_ncnn.py`
  里用 `.clone()` 让 Mat 拥有自有内存。
- 选型结论与实测帧率见 `docs/模型性能测试记录.md`（实时首选 320，≈18FPS）。
