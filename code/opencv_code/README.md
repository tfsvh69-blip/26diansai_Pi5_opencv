# opencv_code — 纯 OpenCV 钢珠检测（不用 YOLO）

## 为什么有这个目录

`task_code/v1.0.py` 用 YOLO11n NCNN 检测小钢珠，帧率只有 ~17FPS（树莓派 5 上 NCNN 推理
就占了大部分时间）。钢珠是 **1cm 金属镜面球 + 固定摄像头桌面场景**，不需要通用目标检测
——纯传统视觉就能做到：

- **检测精度**：离线 12 帧样本 12/12（100%），实时 149/150（99.3%）
- **处理速度**：~11ms/帧（~89FPS），远超摄像头 30FPS 上限；即相机满帧无压力
- **串口通信**：完全复用 `task_code/protocol.py` 格式 `$BALL,...`，下位机无需任何改动

## 文件

| 文件 | 作用 |
|---|---|
| `ball_detector.py` | 钢珠检测核心，唯一事实来源（HoughCircles + 金属镜面确认 + 打分）
| `config.py` | 检测参数持久化到 `detector_config.json`（v1.0.py 启动自动载入）
| `tune_detector.py` | 滑条交互调参工具，调好按 s 保存到 detector_config.json
| `v1.0.py` | 正式主程序：实时检测球 + 把球心 X/Y 经串口发给下位机

## 运行方式

```bash
# 1. 先调好检测参数（在有显示器的桌面会话里）：
/home/hao/vision_env/bin/python3 code/opencv_code/tune_detector.py

# 2. 正式跑检测 + 串口（可以 headless，但建议桌面会话方便看画面）：
/home/hao/vision_env/bin/python3 code/opencv_code/v1.0.py

# 3. 只跑检测不看串口（纯视觉调试）：
/home/hao/vision_env/bin/python3 code/opencv_code/v1.0.py --no-serial

# 4. 串口放宽到任意 CH340（换口临时调试）：
/home/hao/vision_env/bin/python3 code/opencv_code/v1.0.py --any
```

按键（v1.0.py）：`u` = 切换去畸变/原始   `q`/`ESC` = 退出

## 算法

```
画面 BGR
  → Gray + medianBlur(5)
  → HoughCircles (dp=1.2, minDist=20, param1=120, param2=30, minR=10, maxR=18)
  → 每个候选圆打分：
      1. 圆内 V 最大值必须 ≥ min_vmax(205) —— 保证有镜面高光
      2. 圆内 V 标准差必须 ≥ min_vstd(18) —— 保证内部有局部明暗对比
      3. 得分 = v_std + 40×高光占比 + 0.5×max(0, 内亮于环差值)
      得分 ≤ 0 的直接丢弃
  → 按分降序返回候选；主目标 = 分最高的那个
```

核心思路：钢珠是 **镜面金属球**，它的画面上一定有：
1. **清晰的圆形边界**（HoughCircles 基本不会漏）
2. **极端局部对比度** —— 同时有镜面高光（V ≈ 255）和暗边轮廓（V 远低）
3. **孤立的小高亮区域** —— 木纹桌面、Arduino 板等背景没有这种"局部强高光+暗环"

纯颜色/HSV 阈值不可行（钢珠镜面反射环境色），但 Hough 圆 + 金属确认打分在固定场景下足够可靠。

## 迭代记录

- **Hough 初选**（param2=30）= 每帧 ~1 个候选，钢珠几乎必中
- **假圆识别**：木纹上出现的假圆主要是 r=21 大圆（被 `max_radius=18` 杀掉）和 Arduino
  橙色圆形接口（高光确认后分数远低于钢珠）
- **半径范围收紧**：从 10~22 收到 10~18，离线命中率从 8/12 → 12/12
- 实时跑 150 帧：149 命中，94.3% 帧有主目标，x 抖动 1.7px，y 抖动 12.8px（Hough
  在 13px 半径球上的正常噪声，更换场景后可用 tune_detector.py 微调）

## 注意事项

- 换了相机离桌面远近 → 重调 `min_radius`/`max_radius`（用 `tune_detector.py`）
- 换了光照环境 → 重新跑 `tune_camera.py` 固定曝光，然后微调检测参数
- 检测器只看到一个钢珠时直接就是主目标；多个球时选分最高的（面积不作为唯一依据）
