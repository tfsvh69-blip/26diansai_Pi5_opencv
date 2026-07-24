# record_code — 数据采集

为训练模型采集视频/图像的脚本，与 `code/ready_code`（标定/预览）分开放，避免混乱。

## record_dataset.py

用同一个摄像头（固定曝光/增益/白平衡，640x480，MJPG）录制训练数据。
一次运行只产出**一个** MP4：空格键开/关录制，可反复切换多段，全部拼进同一个文件。

```bash
# 输出到 code/record_code/output/dataset_<时间戳>.mp4
/home/hao/vision_env/bin/python3 code/record_code/record_dataset.py

# 或自定义输出路径
/home/hao/vision_env/bin/python3 code/record_code/record_dataset.py my_video.mp4
```

按键：
- **空格**：开始 / 暂停录制（暂停的部分不写入文件）
- **u**：切换 去畸变 / 原始画面（**默认去畸变**）
- **q / ESC**：退出并保存

说明：
- 需在**有显示器/桌面**的会话里跑（有 `cv2.imshow` 窗口），纯 SSH 无 X 转发会失败。
- **默认去畸变**：有 `code/ready_code/camera_calib.npz` 就默认录去畸变画面（与 v1.2 一致，
  部署时同样喂去畸变画面，训练/推理几何一致）；没有标定文件则退回原始画面。
- 写入文件的是干净画面帧，屏幕上的 REC/时长提示不会进视频。
- 复用 `code/ready_code/camera_common.py` 的固定相机参数，改参数只改那一处。
- `output/` 已在 `.gitignore` 中忽略（视频属数据，不入库）。
