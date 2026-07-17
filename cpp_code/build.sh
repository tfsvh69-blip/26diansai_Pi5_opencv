#!/usr/bin/env bash
# 不用 CMake 的一行编译方案（依赖 pkg-config + libopencv-dev）
# 用法: ./build.sh   然后 ./undistort_view
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 undistort_view.cpp -o undistort_view $(pkg-config --cflags --libs opencv4)
echo "✅ 编译完成: ./undistort_view"
