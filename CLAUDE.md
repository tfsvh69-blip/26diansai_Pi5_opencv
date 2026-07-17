# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

树莓派 5 上的电赛视觉项目。一组独立的 Python 脚本，围绕同一个 USB 摄像头
（`Integrated_Webcam_HD`, USB ID `0c45:64ab`, `/dev/video0`）做采集、调参、相机标定与去畸变。
没有构建系统、没有测试框架，每个脚本直接运行。

## 运行环境

固定用这个 venv（系统 Python 里没装 opencv/numpy）：

```bash
/home/hao/vision_env/bin/python3 ready_code/<脚本>.py
```

- OpenCV 4.13 + numpy 2.4，仅此两个依赖。
- 用 `v4l2-ctl`（来自系统包 `v4l-utils`）设置摄像头控制项。
- 带 `cv2.imshow` 窗口的脚本（除引用外几乎都是）**必须在有显示器/桌面的会话里跑**，
  纯 SSH 无 X 转发会失败。

## 各脚本用途

| 脚本 | 作用 |
|---|---|
| `ready_code/v1.0.py` | 自动曝光/自动白平衡版本，保留作对比基线 |
| `ready_code/v1.1.py` | 固定曝光/增益/白平衡的正式预览版本 |
| `ready_code/v1.2.py` | 在 v1.1 基础上载入标定，运行时按 `u` 切换 原始/去畸变；有标定则默认去畸变 |
| `ready_code/tune_camera.py` | 滑条交互式调参工具，退出时打印最终参数值 |
| `ready_code/calibrate_camera.py` | 一键相机标定：空格拍摄→回车自动解算→存 `camera_calib.npz` |
| `ready_code/camera_common.py` | 公共模块（见下），不单独运行 |

标定流程：先跑 `calibrate_camera.py`（拍 15~20 张不同角度/位置，至少 8 张）生成
`ready_code/camera_calib.npz`，之后 `v1.2.py` 会自动载入。采集的原图存在 `ready_code/calib_shots/`。

## 架构要点（需要跨文件理解的部分）

- **`camera_common.py` 是相机参数与标定 I/O 的唯一事实来源。** 固定的曝光/增益/白平衡/亮度
  常量、开摄像头流程（`open_camera`）、标定读写（`save/load_calibration`）、去畸变映射
  （`build_undistort_maps`）都在这里。`calibrate_camera.py` 和 `v1.2.py` 都 `import camera_common as cc`
  复用它。**改相机参数只改这一处**，不要在各脚本里各写一份。v1.0/v1.1 是早期版本，自带一份副本，
  未接入公共模块。
- **版本演进方向：v1.0（自动）→ v1.1（固定参数）→ v1.2（固定参数 + 去畸变）。** 新功能在最新版上加，
  旧版保留作对比。
- **标定板参数写死在 `camera_common.py`：** `CHESSBOARD_SIZE=(9,6)` 内角点、`SQUARE_SIZE_MM=25`。
  对应 `ready_code/chessboard_a4.png`（A4 打印，`findChessboardCorners((9,6))` 实测为 True）。
- **去畸变用预计算映射 + `cv2.remap()`**（不是每帧 `cv2.undistort()`），为树莓派实时性能。
  写新的视觉算法脚本时，按此模式接入：`cc.open_camera()` 开摄像头，
  `cc.load_calibration()` + `cc.build_undistort_maps()` 拿去畸变映射。

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

本机任何系统级配置、服务、脚本部署、依赖安装的变更，都必须追加记录到
`/home/hao/Desktop/系统配置记录.md` 的"后续变更记录"部分（见上级 `Desktop/CLAUDE.md`），
尽量附上关键路径和内容摘要，而不只是写"做了什么"。
