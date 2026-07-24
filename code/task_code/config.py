#!/usr/bin/env python3
"""
任务配置读写 - v1.0.py 的可持久化设置
环境: /home/hao/vision_env/bin/python3

把运行时会调整的设置（YOLO 规格 / 是否去畸变 / 置信度阈值）存到 task_config.json，
运行时改了自动写回，下次启动自动沿用。配置文件与本模块同目录：
    code/task_code/task_config.json   （首次运行自动创建；已在 .gitignore 忽略，属本机状态）

用法：
    import config
    cfg = config.load()          # 读，缺文件/损坏都退回 DEFAULTS
    cfg["size"] = 416
    config.save(cfg)             # 原子写回
"""

import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task_config.json")

# 会被持久化的配置项及其默认值
DEFAULTS = {
    "size": 320,          # YOLO 规格：320 / 416 / 640
    "undistort": True,    # 是否默认去畸变（有标定时生效）
    "conf": 0.25,         # 置信度阈值
}
VALID_SIZES = (320, 416, 640)


def load():
    """读配置；文件缺失或损坏都安全退回 DEFAULTS，并对取值做兜底校验。"""
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in DEFAULTS:
                    if k in data:
                        cfg[k] = data[k]
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # 损坏就用默认值，不让脚本崩

    # 兜底校验，防止手改坏了配置
    if cfg["size"] not in VALID_SIZES:
        cfg["size"] = DEFAULTS["size"]
    try:
        cfg["conf"] = float(cfg["conf"])
    except (TypeError, ValueError):
        cfg["conf"] = DEFAULTS["conf"]
    cfg["undistort"] = bool(cfg["undistort"])
    return cfg


def save(cfg):
    """原子写回配置（只写 DEFAULTS 里认识的键）。成功返回 True。"""
    data = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
    try:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CONFIG_FILE)   # 原子替换，避免写一半损坏
        return True
    except OSError:
        return False


if __name__ == "__main__":
    print("配置文件:", CONFIG_FILE)
    print("当前内容:", load())
