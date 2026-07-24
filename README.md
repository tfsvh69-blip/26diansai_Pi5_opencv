# 26diansai_Pi5_opencv

树莓派 5 上的电赛视觉项目：围绕一个 USB 摄像头（`Integrated_Webcam_HD`, USB `0c45:64ab`,
`/dev/video0`）做**稳定采集、参数调校、相机标定与实时去畸变**，并在其上跑 **YOLO11n NCNN
球检测**、把球心坐标经 **USB 串口**发给下位机（单片机）的一组独立 Python 脚本。

核心思路：关闭自动曝光/自动增益/自动白平衡，改用现场调出的**固定参数**，保证 OpenCV / 检测
拿到的画面在光线变化、物体移动时颜色与亮度一致；再用棋盘格做相机标定，运行时可自由切换
「原始画面 / 去畸变画面」；检测与部署统一喂去畸变画面，几何一致。

## 目录结构

```
code/ready_code/   相机基础：调参 / 标定 / 预览 / 公共模块（camera_common）
code/record_code/  数据采集录像（训练数据）
code/detect_code/  YOLO11n NCNN 检测与三规格性能基准
code/task_code/    正式题目：检测球 + 串口发坐标（从 v1.0 起编号，含热插拔与通信协议）
code/cpp_code/     C++ OpenCV 去畸变工程骨架
models/            三规格 NCNN 模型（best_ncnn_320/416/640）
docs/              系统操作记录、模型性能测试记录
reference/         电赛题目算法参考资料
```

各子目录另有 README 说明；检测选型见 `docs/模型性能测试记录.md`，
通信格式见 `code/task_code/README.md`。下面是相机基础部分。

## 环境

```bash
/home/hao/vision_env/bin/python3 code/ready_code/<脚本>.py
```

- 依赖：OpenCV 4.13 + numpy（venv 内），系统包 `v4l-utils`（提供 `v4l2-ctl`）。
- 带窗口预览的脚本需在有显示器/桌面的会话运行（纯 SSH 无 X 转发不行）。

## 脚本一览

| 脚本 | 作用 |
|---|---|
| `code/ready_code/v1.0.py` | 自动曝光/自动白平衡版本，作对比基线 |
| `code/ready_code/v1.1.py` | 固定曝光/增益/白平衡的正式预览版本 |
| `code/ready_code/v1.2.py` | v1.1 + 相机去畸变，运行时按 `u` 切换 原始/去畸变（有标定则默认去畸变） |
| `code/ready_code/tune_camera.py` | 滑条交互式调参，退出时打印最终参数值 |
| `code/ready_code/calibrate_camera.py` | 一键标定：空格拍摄 → 回车自动解算 → 存 `camera_calib.npz` |
| `code/ready_code/camera_common.py` | 公共模块：固定参数 / 开摄像头 / 标定读写 / 去畸变映射（唯一事实来源） |

## 使用流程

1. **调参**（换环境/光照时）：跑 `tune_camera.py`，把打印出的最终值填回 `camera_common.py`。
2. **标定**：跑 `calibrate_camera.py`，拿 9×6 内角点、25mm 方格的棋盘（`code/ready_code/chessboard_a4.png`
   A4 打印）在镜头前，拍 15~20 张不同角度/位置，回车自动解算，生成 `code/ready_code/camera_calib.npz`。
3. **使用**：跑 `v1.2.py`，默认去畸变画面，按 `u` 随时切回原始对比。

写新的视觉算法脚本时，`import camera_common as cc`，用 `cc.open_camera()` 开摄像头，
`cc.load_calibration()` + `cc.build_undistort_maps()` 拿去畸变映射即可复用同一套参数。

## 硬件要点

- 用 `v4l2-ctl` 直接下发 V4L2 控制项（`exposure_time_absolute` / `white_balance_temperature`
  在 OpenCV 属性映射里不可靠）；设固定曝光/白平衡前必须先切手动模式。
- 定焦镜头，无自动对焦，清晰度靠物理调整距离。
- MJPG 模式仅支持 1280x720 / 640x480 两档，固定参数版用 640x480 以提高帧率、减少运动模糊。
