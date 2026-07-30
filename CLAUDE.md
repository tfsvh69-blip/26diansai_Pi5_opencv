# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

树莓派 5 上的电赛视觉项目。一组独立的 Python 脚本，围绕同一个 USB 摄像头
（`Integrated_Webcam_HD`, USB ID `0c45:64ab`, `/dev/video0`）做采集、调参、相机标定与去畸变，
并在其上跑 YOLO11n NCNN 球检测、把球心坐标经 USB 串口发给下位机（单片机）。
没有构建系统、没有测试框架，每个脚本直接运行。

代码按类别分目录：`code/ready_code`（相机基础：调参/标定/预览/公共模块）、
`code/record_code`（数据采集录像）、`code/detect_code`（NCNN 检测与性能基准）、
`code/task_code`（正式题目：YOLO检测+串口，从 v1.0 起编号）、
`code/opencv_code`（纯 OpenCV 钢珠检测，不用 YOLO，帧率 89FPS）、
`code/cpp_code`（C++ 去畸变骨架）；
模型在 `models/`、文档在 `docs/`、参考资料在 `reference/`。

## 运行环境

固定用这个 venv（系统 Python 里没装 opencv/numpy）：

```bash
/home/hao/vision_env/bin/python3 code/ready_code/<脚本>.py
```

- OpenCV 4.13 + numpy 2.4 + Pillow 12.3（中文渲染用）。
- 用 `v4l2-ctl`（来自系统包 `v4l-utils`）设置摄像头控制项。
- 带 `cv2.imshow` 窗口的脚本（除引用外几乎都是）**必须在有显示器/桌面的会话里跑**，
  纯 SSH 无 X 转发会失败。

## 各脚本用途

| 脚本 | 作用 |
|---|---|
| `code/ready_code/v1.0.py` | 自动曝光/自动白平衡版本，保留作对比基线 |
| `code/ready_code/v1.1.py` | 固定曝光/增益/白平衡的正式预览版本 |
| `code/ready_code/v1.2.py` | 在 v1.1 基础上载入标定，运行时按 `u` 切换 原始/去畸变；有标定则默认去畸变 |
| `code/ready_code/tune_camera.py` | 滑条交互式调参工具，退出时**自动把最终参数写回 `camera_common.py`**（并打印） |
| `code/ready_code/calibrate_camera.py` | 一键相机标定：空格拍摄→回车自动解算→存 `camera_calib.npz` |
| `code/ready_code/camera_common.py` | 公共模块（见下），不单独运行 |
| `code/detect_code/yolo_ncnn.py` | YOLO11n NCNN 推理封装（letterbox/推理/解码/NMS），被 detect_live/benchmark 复用 |
| `code/detect_code/detect_live.py` | 载入 NCNN 模型实时检测预览（复用固定参数 + 默认去畸变） |
| `code/detect_code/benchmark.py` | 对 320/416/640 三规格测纯推理耗时与端到端 FPS，输出对比 |
| `code/task_code/v1.0.py` | **正式题目主程序**：实时检测球 + 把主目标球心 X/Y 经串口发给下位机；摄像头/串口双热插拔 |
| `code/task_code/v1.1.py` | 在 v1.0 基础上加 **BoT-SORT(无ReID)+GMC+卡尔曼滤波**跟踪，保证主目标框选连续性（漏检/遮挡靠预测续上、不因别的球更大而跳目标），最后过 One Euro Filter 平滑坐标再发串口 |
| `code/task_code/bot_sort.py` | BoT-SORT 轻量实现：ByteTrack 式高/低置信度两级关联 + 稀疏光流 GMC + (cx,cy,w,h) 卡尔曼滤波 + 手写匈牙利分配（不依赖 scipy/lap/ultralytics，本机未装），不单独运行 |
| `code/task_code/one_euro_filter.py` | One Euro Filter 一维平滑滤波器，供 v1.1 对主目标像素坐标做最后平滑，不单独运行 |
| `code/task_code/serial_test.py` | 串口收包监视：打印下位机发来的数据，验证链路（支持热插拔） |
| `code/task_code/serial_link.py` | 串口链路封装：锁定固定物理 USB 口 + 自动等待/断线重连，不单独运行 |
| `code/task_code/camera_link.py` | 摄像头链路封装：`open_camera` 外包一层等待/掉线重连，不单独运行 |
| `code/task_code/protocol.py` | 上位机→下位机通信格式定义（`$BALL,found,x,y,n*CHK`），唯一事实来源 |
| `code/opencv_code/ball_detector.py` | 纯 OpenCV 钢珠检测核心 v2：HoughCircles + 多因素综合评分 + 速度预测跟踪 + 置信度衰减，~89FPS |
| `code/opencv_code/v1.0.py` | 纯 OpenCV 版正式主程序（替代 YOLO）：检测+串口，复用同一套 camera_link/serial_link/protocol |
| `code/opencv_code/v1.1.py` | 在 v1.0 基础上把 `ball_detector.py` 的候选圆接入 `task_code/bot_sort.py` 的 BoT-SORT(无ReID)+GMC+卡尔曼滤波，解决"能测到但不连续/多候选间跳目标"的问题，最后过 One Euro Filter 再发串口；不改 v1.0.py / ball_detector.py 一行代码 |
| `code/opencv_code/tune_detector.py` | 钢珠检测参数调参工具：中文滑条 + 实时视觉反馈（候选/淘汰/主目标 + 拒绝原因） |
| `code/opencv_code/config.py` | opencv_code 的配置读写（检测参数 + 去畸变偏好 → detector_config.json） |

标定流程：先跑 `calibrate_camera.py`（拍 15~20 张不同角度/位置，至少 8 张）生成
`code/ready_code/camera_calib.npz`，之后 `v1.2.py` 会自动载入。采集的原图存在 `code/ready_code/calib_shots/`。

## 架构要点（需要跨文件理解的部分）

- **`camera_common.py` 是相机参数与标定 I/O 的唯一事实来源。** 固定的曝光/增益/白平衡/亮度
  常量、开摄像头流程（`open_camera`）、标定读写（`save/load_calibration`）、去畸变映射
  （`build_undistort_maps`）都在这里。`calibrate_camera.py`、`v1.2.py`、`record_dataset.py`、`detect_code/*` 都 `import camera_common as cc`
  复用它。**改相机参数只改这一处**，不要在各脚本里各写一份。改的方式：跑 `tune_camera.py`
  调好后退出，会自动写回这里的四个常量。v1.1/v1.2 也 `import camera_common` 复用同一份阈值；
  只有 v1.0 仍是早期自动曝光基线、自带一份副本。
- **版本演进方向：v1.0（自动）→ v1.1（固定参数）→ v1.2（固定参数 + 去畸变）。** 新功能在最新版上加，
  旧版保留作对比。
- **标定板参数写死在 `camera_common.py`：** `CHESSBOARD_SIZE=(9,6)` 内角点、`SQUARE_SIZE_MM=25`。
  对应 `code/ready_code/chessboard_a4.png`（A4 打印，`findChessboardCorners((9,6))` 实测为 True）。
- **去畸变用预计算映射 + `cv2.remap()`**（不是每帧 `cv2.undistort()`），为树莓派实时性能。
  写新的视觉算法脚本时，按此模式接入：`cc.open_camera()` 开摄像头，
  `cc.load_calibration()` + `cc.build_undistort_maps()` 拿去畸变映射。
- **约定（后续所有脚本默认遵循）：**
  1. **默认去畸变**——处理/采集/预览一律用去畸变后的画面（有 `camera_calib.npz` 就默认开，
     没有再退回原始）。理由：部署与训练都喂去畸变画面，几何一致。`v1.2.py` 和
     `code/record_code/record_dataset.py` 已按此实现，新脚本照做。
  2. **固定阈值只认 `camera_common.py` 当前值**——曝光/增益/白平衡/亮度以 `camera_common.py`
     里此刻的四个常量为准（现为 tune 后的最新值），不要在脚本里另写。需要改就跑
     `tune_camera.py` 重新调，退出自动写回。

## 检测 + 串口任务（`code/task_code`，从 v1.2 起）

- **NCNN 推理走 `code/detect_code/yolo_ncnn.py` 封装。** 模型是 YOLO11n 单类 `ball`，
  导出的 `out0` 是 `[5,N]`=`[cx,cy,w,h,置信度]`，框坐标已是输入尺度像素、已过 Sigmoid
  （DFL/anchor 解码烘焙进图），后处理只需阈值→xywh转xyxy→反 letterbox→NMS，**不用手写 anchor 解码**。
  实时规格选型见 `docs/模型性能测试记录.md`（首选 320，≈17FPS）。
- **踩坑：`ncnn.Mat(numpy)` 只包装不拷贝缓冲区**，预处理数组出作用域会段错误，`yolo_ncnn.py`
  用 `.clone()` 让 Mat 拥有自有内存。
- **`YoloNcnn.decode()` 支持临时覆盖置信度阈值**（`decode(..., conf_thres=x)`，不传则用
  `self.conf_thres`），是给 v1.1 的 ByteTrack 式高/低两级关联用的：一次前向，按低阈值解码
  一次拿到全部候选框，再由 `bot_sort.py` 内部按高/低两级阈值拆分，不用重复推理。
- **通信格式唯一事实来源是 `code/task_code/protocol.py`。** 帧格式
  `$BALL,<found>,<x>,<y>,<n>*<CHK>\r\n`（NMEA 风格 ASCII + XOR 校验），
  坐标是去畸变 640x480 画面像素、原点左上。改格式只改这一处，下位机解析同步。
  **主目标定义随版本略有差异**：v1.0 每帧独立选面积最大的球；v1.1 起改成"锁定跟踪 id"——
  只要该轨迹没有真正丢失就一直是同一个目标（漏检/遮挡靠卡尔曼预测续上），
  轨迹消失了才按面积最大重新挑，避免逐帧选最大导致的目标跳变。
- **v1.1 起加入目标跟踪，保证框选连续性**：`code/task_code/bot_sort.py` 实现
  BoT-SORT 的无 ReID 版本（ByteTrack 两级关联 + 相机运动补偿 GMC + 用 (cx,cy,w,h)
  建模的卡尔曼滤波），`code/task_code/one_euro_filter.py` 对跟踪后的主目标坐标做最后
  一道平滑再发串口。ReID（外观特征）按要求关闭：单类球没有可辨识外观，且树莓派5
  跑 ReID 网络划不来；本机也没装 scipy/lap/ultralytics，所以匈牙利分配和卡尔曼滤波
  都是纯 numpy 手写，不是套的库。写新的多目标跟踪场景可复用 `bot_sort.BotSort`。
- **热插拔封装：`serial_link.py` / `camera_link.py`** 各自把"找设备/等设备/断线重连"收在一处，
  串口锁定固定物理 USB 口（`/dev/serial/by-path/...`，换口改 `PREFERRED_BYPATH`），
  摄像头掉线时 `v1.0.py`/`v1.1.py` 显示占位画面、插回自动续，两条链路都不因拔插崩溃。
  写新任务脚本复用它们。

## 硬件约束（决定了代码为什么这么写）

- **用 `v4l2-ctl` 下发控制项，而非只用 `cap.set()`**：`exposure_time_absolute` /
  `white_balance_temperature` 在 OpenCV 属性映射里不可靠。
- **设置固定曝光/白平衡前必须先切手动模式**：`auto_exposure=1`、`white_balance_automatic=0`，
  否则相关控制项处于 inactive，设了也不生效。
- **定焦镜头，无自动对焦硬件**：清晰度靠物理调整距离，代码不提供对焦。
- **MJPG 模式只有 1280x720 / 640x480 两档分辨率**（无 640x360），固定参数版用 640x480。
  用 MJPG 而非默认 YUYV 以提高帧率、减少运动模糊。
- 换了光照环境后需重新跑 `tune_camera.py` 更新固定参数，并重新标定。

## 变更记录要求

本机任何系统级配置、服务、脚本部署、依赖安装的变更，都必须追加记录到**两处**：

1. **全局记录**：`/home/hao/Desktop/系统配置记录.md` 的"后续变更记录"部分
   （见上级 `Desktop/CLAUDE.md`），全机器视角。
2. **本项目本地记录**：[docs/系统操作记录.md](docs/系统操作记录.md)，26diansai 项目视角，
   方便在项目内直接查阅。凡是与本项目相关、涉及系统/桌面环境层面的操作（开机自启动、
   桌面配置、设备权限、依赖安装、涉及登录会话的改动等）都要在这里追加一条。

两处都要写清**动了哪些文件/配置、为什么、怎么验证、怎么回滚**，
尽量附上关键路径和内容摘要，而不只是写"做了什么"。
