#!/usr/bin/env bash
# 开机自启动包装脚本：等待摄像头就绪后，启动 C++ 去畸变预览窗口。
# 由 ~/.config/autostart/camera-undistort.desktop 在桌面登录后调用。
#
# 解决两个开机自启动的坑：
#   1) undistort_view 用相对路径读 camera_calib.yaml，必须先 cd 到本目录；
#   2) 开机瞬间 /dev/video0 可能还没枚举好，直接启动会打不开摄像头而退出，
#      这里先等待设备就绪再启动。
set -u

cd "$(dirname "$0")" || exit 1

LOG="$HOME/.local/state/camera-autostart.log"
mkdir -p "$(dirname "$LOG")"
# 保留最近一次日志（覆盖），避免无限增长
exec >"$LOG" 2>&1
echo "==== $(date '+%F %T') autostart 触发 ===="
echo "  工作目录: $(pwd)"
echo "  DISPLAY=${DISPLAY:-<未设置>}"

# 等待摄像头设备就绪（最多 30 秒）
for i in $(seq 1 30); do
    if [ -e /dev/video0 ]; then
        echo "  /dev/video0 就绪 (等待 ${i}s)"
        break
    fi
    sleep 1
done

if [ ! -e /dev/video0 ]; then
    echo "  ❌ 超时 30s 仍无 /dev/video0，放弃启动"
    exit 1
fi

# 再给 X server / 摄像头一点稳定时间
sleep 2

echo "  启动 undistort_view ..."
exec ./undistort_view
