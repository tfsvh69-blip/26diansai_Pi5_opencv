#!/usr/bin/env python3
"""
opencv_code 任务配置读写 —— 检测器参数 + 是否去畸变 的持久化
环境: /home/hao/vision_env/bin/python3

把现场会调的东西（BallDetector 的 Hough/金属确认参数、是否默认去畸变）存到
detector_config.json，tune_detector.py 调好保存、v1.0.py 启动载入，换场景免重调。
    code/opencv_code/detector_config.json  （首次运行自动创建，属本机状态，已 .gitignore）

用法：
    import config
    cfg = config.load()          # {"undistort": True, "detector": {...}}
    config.save(cfg)
"""

import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector_config.json")

# detector 为空 dict 时 BallDetector 用其内置默认值（离线迭代选定的值）
DEFAULTS = {
    "undistort": True,     # 项目约定：默认去畸变（有 camera_calib.npz 时生效）
    "detector": {},        # BallDetector.as_dict() 的内容；空=用内置默认
}


def load():
    cfg = {"undistort": DEFAULTS["undistort"], "detector": dict(DEFAULTS["detector"])}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg["undistort"] = bool(data.get("undistort", cfg["undistort"]))
                if isinstance(data.get("detector"), dict):
                    cfg["detector"] = data["detector"]
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # 损坏就用默认，不让脚本崩
    return cfg


def save(cfg):
    """原子写回。成功返回 True。"""
    data = {
        "undistort": bool(cfg.get("undistort", DEFAULTS["undistort"])),
        "detector": cfg.get("detector", {}) or {},
    }
    try:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CONFIG_FILE)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    print("配置文件:", CONFIG_FILE)
    print("当前内容:", load())
