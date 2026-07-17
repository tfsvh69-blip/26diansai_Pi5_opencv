# cpp_code — C++ 版 OpenCV 去畸变画面

树莓派 5（aarch64）上的 C++ OpenCV 工程骨架，对应 Python 端 `ready_code/v1.2.py`：
固定曝光/增益/白平衡打开摄像头，载入标定数据，运行时按 `u` 切换 原始/去畸变 画面。

## 依赖（系统级，需一次性安装）

```bash
sudo apt update
sudo apt install -y libopencv-dev pkg-config cmake
```

- `libopencv-dev`：C++ 头文件和库（apt 版本 4.6.0）。注意：venv 里的 pip OpenCV(4.13)
  只有 Python 绑定，**不提供 C++ 开发文件**，所以必须装这个系统包。
- 标定数据 `camera_calib.yaml` 由 Python 端 `camera_calib.npz` 导出，跨 OpenCV 版本通用。

## 编译

两种方式，任选其一：

```bash
# 方式一：CMake（推荐，标准工程）
mkdir -p build && cd build && cmake .. && make && cd ..
cp build/undistort_view .

# 方式二：一行编译（依赖 pkg-config）
./build.sh
```

## 运行

```bash
./undistort_view
```

需在有显示器/桌面的会话运行（要弹预览窗口，纯 SSH 无 X 转发不行）。
运行时 `camera_calib.yaml` 需在当前工作目录。

按键：`u` 切换 原始/去畸变，`q`/`ESC` 退出。有标定时默认显示去畸变画面。

## 与 Python 端的关系

- 固定相机参数以 `ready_code/camera_common.py` 为唯一事实来源；本目录 `undistort_view.cpp`
  里的常量是它的副本，改参数需两边同步（或重跑 `tune_camera.py` 后一起更新）。
- 重新标定后，用 Python 把新的 `camera_calib.npz` 再导出成本目录的 `camera_calib.yaml`。
