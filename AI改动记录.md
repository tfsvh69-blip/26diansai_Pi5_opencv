# AI 改动记录（rollball 及相关）

> 这份文档记录 Claude 每次帮我改了什么、为什么、怎么验证、怎么回滚，方便我隔一段时间回来能快速想起来。
> **最新的记在最上面。** 只记"改动"，不记闲聊。涉及的主要目录：`Formal_code/rollball_code/`（滚球视觉+串口）。

---

## 2026-08-01 —— 修录像 1x 播放比真实快 ~1.67 倍（0.6x 才贴合）：USB 相机突发性导致预热测帧率虚高

**动机**：用户上机试用网页回放后发现"录像 1x 播放比真实世界快，要 0.6x 才贴切"。已确认两条证据：录像时终端**没有**"⚠️ 录像写盘跟不上丢帧"告警（排除编码丢帧）、一段录像真实 START→STOP 约 5 秒。根因锁定：录像 fps 是在**开录后前 0.5s 预热窗口**里用帧到达时间戳测的（`mcu_link._VideoRecorder._finish_warmup`），USB 相机帧是"攒一批→一次吐一批"的突发式到达，窗口正好撞上突刺就把 fps 测虚高（实测元数据 49~53fps、真实 ~30fps，即 1.67 倍）。

**改了哪些文件**：仅 `Formal_code/rollball_code/mcu_link.py`（`_VideoRecorder`，录像写盘后台线程）。

1. **预热窗口拉长**：`WARMUP_MIN_FRAMES` 15→**60**、`WARMUP_MIN_TIME` 0.5→**2.0s**、`WARMUP_MAX_FRAMES` 60→**150**。2 秒窗口跨多个突刺周期取平均，从根上压低突发性影响（代价：录像开头最多延迟 2s 才开始写盘，但帧全部在内存缓冲、不丢，实测 2s 缓冲峰值 ~100 帧/92MB，Pi 5 无压力）。
2. **收尾重封装校正（双保险）**：worker 循环新增 `rec_meta = {"count","first","last","used_fps"}` 记录整段录像的帧数/首末帧墙钟/封装 fps（每段 open 重置）；`_maybe_reencode` 在每段收尾（close/quit/重叠 open）时用**整段真实时长**算真实平均 fps，与封装 fps 偏差 **>8%** 就 `_reencode_fps` 读回 mp4、按真实 fps 重写一份再 `os.replace` 原子替换。即使突刺周期 >2s、预热仍偏差，收尾也兜底，1x 精确贴合真实时间。失败保留原文件 + 告警，不崩。
3. **`_reencode_fps` 临时文件放 `video_dir/_reencode_tmp/` 子目录**（保留 `.mp4` 后缀——实测 OpenCV VideoWriter 对无后缀/陌生后缀**打不开**；子目录不会被网页录像列表 `os.listdir` 扫到），成功后原子替换 + 清临时目录。

**效果/验证**（无相机，用合成帧直接驱动 `_VideoRecorder` 模拟真实节奏，三个用例全绿）：
- **突刺开录**（前 2 批很密、之后 ~27fps）：预热测 45.2fps（虚高），收尾自动校正→34.2fps，文件时长 9.06s vs 实际 9.23s（偏差 <2%），日志打"📼 录像帧率校正: 45.2→34.2fps"。
- **平稳 30fps**：2s 预热测准 29.9fps，无需校正，5.01s vs 5.02s。
- **短录像提前 STOP**（1.2s，预热被截断）：1.20s vs 1.20s。
- `py_compile` 三文件通过；网页回放链路自测（`/tmp/replay_selftest.py`）重跑无回归。
- **上机验证留给用户**：再录一段，网页回放 1x 应贴合真实速度；终端看是否打印"📼 录像帧率校正"（仅当预热仍偏差时出现，属正常）。

**回滚**：`Formal_code/` 未纳入 git，需手动还原 `mcu_link.py`（预热常量改回 15/0.5/60，删 `rec_meta` 线程/`_maybe_reencode`/`_reencode_fps`/`_cleanup_tmpdir`，`_finish_warmup`/`_open_writer` 去掉 `rec_meta` 参数，三处收尾去掉 `_maybe_reencode` 调用）。自测脚本 `/tmp/recorder_selftest.py`。

---

## 2026-08-01 —— 网页端回放录像（MJPEG 自建播放器，零新依赖）

**动机**：用户希望网页端（:8080 那个页面）能直接看录像回放，而不是拷走 mp4 用本地播放器。现状是网页只有实时流；录像由单片机 TASK START/STOP 触发写 `Formal_code/mp4/*.mp4`，编码 **FMP4（MPEG-4 Part 2）**——现代浏览器 `<video>` 原生不支持，直接播会黑屏；Pi 上也没装 ffmpeg。用户选定方案：**复用现有网页服务 + 就播现有录像画面（不改录像逻辑）+ MJPEG 自建播放器**（OpenCV 读 mp4 + 复用 `_Stream` 推流机制，任何浏览器都能看，零新依赖）。

**改了哪些文件**（2 个，均 `Formal_code/rollball_code/` 下；`Formal_code/` 未纳入 git，回滚需手动还原）：

1. **`mjpeg_server.py`**
   - 新类 **`_ReplayReader`**：per-connection 回放读取线程，按单调时钟累加绝对目标时刻 pacing（不漂移），`stop()` 先 join 再 `cap.release()`。
   - 新类 **`_ReplayPlayer`**：列录像（`list_recordings`，按 mtime 倒序）/ 读元数据（`info`，按 `(size,mtime)` 缓存，录制中自动失效）/ 开流（`open_stream`，`CAP_PROP_POS_MSEC` seek + 探测 read 复位，**正在写入的 mp4 moov 未落盘会被探测判成打不开**）/ 单帧静图（`frame_jpeg`）。`_resolve` 做 basename + `.mp4` 校验防路径穿越。
   - **4 个新路由**（`replay_dir` 传入才启用，None 时全部优雅降级）：`GET /api/recordings`（列表）、`GET /api/replay/info?file=`（元数据）、`GET /replay?file=&t=&speed=`（MJPEG 流）、`GET /replay/frame?file=&t=`（单帧 JPEG 静图）。
   - **per-connection 独立 `_Stream`**：每条 `/replay` 连接建自己的 `_Stream`（独享编码线程），多客户端互不串台；`_serve_replay` 复用 `_serve_stream` 作内层推流，finally 停 reader + 流。
   - **`MjpegServer.stop()` 兜底清理**：登记活跃回放连接（`_replay_started/_replay_finished`），`stop()` 时统一停掉 reader 和 per-connection 流——否则主程序退出时还挂着的回放 daemon 线程带着 `cv2.VideoCapture` 被强杀，实测会崩 **SIGABRT（"FATAL: exception not rethrown"）**。
   - `index_html()` 加"回放"按钮 + 面板：录像下拉、播放/暂停、进度条（`onchange` 松手才 seek）、倍速（0.5/1/2/4）、时间标签。前端进度用**墙钟计时**（`t = playStartT + elapsed*speed`，到 duration 自动停），与服务器帧节奏解耦；暂停 = 把 `<img>` src 换成 `/replay/frame` 静图兜底（不依赖旧连接何时断开）。
2. **`v1.1_beta.py`**（主循环**零改动**）
   - 新增常量 `REPLAY_DIR = os.path.join(PROJECT_ROOT, "Formal_code", "mp4")`（定义在 `MjpegServer` 构造之前）。
   - `MjpegServer(port=args.stream_port, replay_dir=REPLAY_DIR)` 接线回放目录。
   - `McuLink(video_dir=REPLAY_DIR)` 改引用同一常量（原 `os.path.join(...)` 两处共用，防以后改目录漏改）。

**关键事实/决策**：
- 录像 fps 是每段实测的非整数（49~54），seek 必须用 `POS_MSEC` 而非按帧号算；OpenCV 能正常回读 + 跳帧（实测 3 个 mp4 全过）。
- 前端"暂停"时浏览器对 `<img>` 换 src 会 abort 旧连接；服务器靠 `wfile.write` 报 `BrokenPipe` 收尾。**http.client 的优雅 close(FIN) 不会立即让服务器写报错**（TCP 写缓冲），reader 会跑到 EOF 才停——这是正常现象，浏览器关标签页是 RST、毫秒级清理（已实测 0.25s）。
- 回放连接不触碰 `main.has_clients`，不会连带拉起直播编码，空闲零开销。

**效果/验证**：
- `py_compile` 两文件通过。
- 独立集成自测脚本（合成 60 帧 160x120 mp4 + http.client / 原始 socket 断言）全绿：`/api/recordings` 倒序、`/api/replay/info` 元数据（fps≈30/frames=60/duration≈2）、路径穿越/不存在被拒、`/replay/frame` 返回 FFD8 JPEG、`/replay` 流读回 multipart 真实帧、**并发两连接不同 t 首帧内容不同（隔离）**、**RST 断连后 reader ≤3s 退出**、`server.stop()` 后无残留 reader 且不崩。
- 用真实录像 `Formal_code/mp4/`（fps 49.65、1276 帧、640x480）重跑同一脚本全过，非整数 fps 的 seek/读取正常。
- HTML 渲染检查：回放面板/JS 注入正确、占位符无残留、JS 花括号平衡。
- `v1.1_beta.py` 模块导入 + 接线断言通过。
- **浏览器手动验证留给用户上机**（需 Pi 上跑 `v1.1_beta.py --source usb --headless` 后手机/电脑开 `http://<IP>:8080/`）：录像下拉倒序、播放/暂停/拖动/倍速/播完自动停、录制中文件提示"正在写入或损坏"、两标签互不干扰、直播流仍正常。

**回滚**：`Formal_code/` 未纳入 git，需手动还原 `mjpeg_server.py`（删两个新类/4 路由/回放面板，`MjpegServer.__init__` 去掉 `replay_dir` 与 `_replay_*` 登记）和 `v1.1_beta.py`（`REPLAY_DIR` 常量、`replay_dir=` 参数、`McuLink` 的 `video_dir` 改回 `os.path.join(...)`）。开发期自测脚本在 `/tmp/replay_selftest.py`（未入库）。

---

## 2026-07-30 —— 网站开关改回默认开启（上一条已证实它不是帧率瓶颈）

**动机**：上一条修好 RealSense 帧率协商 bug 后，用户上机实测**开着网站也一样有 30 多帧**，确认网站开销确实可以忽略，要求改回默认开启，并要求把结论更新进文档。

**改了哪些文件**：仅 `Formal_code/rollball_code/v1.1_beta.py`。
- `--stream`（`store_true`，默认关）改回 `--no-stream`（`store_true`，**默认开**，需要关掉再显式加参数）——即撤销上一条改动里"网站默认关闭"的部分，其余（后台 JPEG 编码线程、`has_clients` 门槛）都不动。
- `run_gui()` 里 `stream_server` 的构造条件从 `if args.stream:` 改回 `if not args.no_stream:`。
- 相关注释更新为记录这次"排查+确认+改回"的完整结论：网站不是瓶颈的原因是 JPEG 编码在后台线程做（`cv2.imencode` 会释放 GIL，能吃到 Pi5 的另一个核）且只有真有客户端连着时才编码，默认开着基本零额外开销。

**效果/验证**：真实硬件跑 `--headless --no-mcu`（未连 MCU），确认：不加任何参数网站就自动启动（打印出访问地址），headless 状态行仍稳定 **30.0FPS**，和网站默认关闭时一致；`py_compile` 通过。

**回滚**：`Formal_code/` 未纳入 git，需手动改回（把 `--no-stream` 换回 `--stream`、`store_true`+条件取反）。

---

## 2026-07-30 —— 找到并修复"只有15fps"的真凶：RealSense 640x480@60fps(bgr8) 根本不存在这个档位

**动机**：用户反馈主循环只有 ~15fps，怀疑是 MJPEG 网站拖慢，想把网站做成默认关闭的开关；并且给出一条关键线索——"有没有人看这个 web，它都只有15帧"。这条线索直接指向网站大概率不是元凶（`MjpegServer` 的 JPEG 编码只在 `has_clients=True` 时才跑，没人看时后台线程本就空闲），同时截图还暴露了另一个真 bug：左右两个预览面板对同一个球显示了不同的 X 坐标。

**本机恰好接了实体 RealSense D435，直接上机测出了真凶（不是猜的）**：
- 查询这颗 D435 支持的流档位发现：**640x480 分辨率下 bgr8 格式压根没有 60fps 这一档**（SDK 只给 6/15/30fps）。`--fps` 默认是 60，`RealSenseCamera` 请求 `640x480@60fps(bgr8)` 必然让 `pipeline.start()` 抛 `Couldn't resolve requests`，代码原有的降级策略是直接问 SDK 要"完全不设限的默认配置"。
- 实测这个"完全不设限"的默认配置协商到的是 **640x480 rgb8 @15fps**，实际交付速率只有 **~12.6fps**（用 `pipeline.try_wait_for_frames` 连续 3 秒实测），和用户报告的 15.9fps 完全对得上。
- 而同一分辨率、格式改成 **bgr8**（不用降级到 rgb8 还得转换）其实是有 **30fps** 这一档的，实测协商后交付速率 **~25.9fps**——比原来快了一倍还多。
- 用真实保存的 `rollball_config.json`（ROI/曝光/detector 参数）跑"读帧+ROI裁剪+detect()"完整管线，实测 **27.6fps**，`detect()` 本身只要 **~1.1ms/帧**——顺带证实 BallDetector 的 Hough 检测从来都不是瓶颈，全程都是相机协商到了一个很差的档位。

**改了哪些文件**：仅 `Formal_code/rollball_code/v1.1_beta.py`。

1. **【真凶修复】`RealSenseCamera.__init__` 的降级策略改成"同分辨率+bgr8，按 fps 从高到低试几档"**：原来"一次尝试失败就跳到完全不设限"的两级策略，改成 `[fps, 30, 15, 6]`（去重保序）依次尝试同分辨率同格式的不同 fps 档位，同一个 `pipeline` 对象可以直接复用重试（实测确认不用每次重建）；全部失败才退到完全不设限的兜底。这样"请求60拿不到"时会先落在"同格式更高fps"而不是随便一个"格式都变了的最低档"。
2. **【新增诊断】`RealSenseCamera` 起流成功后打印实际协商到的分辨率/格式/fps**（`_print_negotiated_profile`），和请求值不一致会带 ⚠️ 标记，方便下次一眼看出是不是又掉到了更差的档位。
3. **【新增诊断】主循环加 `detect_ms_ema`**：`detector.detect()` 单帧耗时的 EMA(毫秒)，显示在 GUI 的 HUD 和 headless 每 2 秒的状态行里，用来区分"慢在相机"还是"慢在检测算法"。
4. **【bug 修复】右侧"高光二值掩膜"面板的坐标标签和左图对不上**：右面板之前用 `ball_detector.draw()` 在 ROI 局部坐标系的子图上画圈+印文字，左面板用自己的 `draw_detections()` 换算成整幅画面绝对坐标再画——同一个球两个面板显示两个不同的数字（用户截图里左"(301,217)"、右"(262,17x)"），看起来像识别出了分歧，其实是同一份检测结果、只是标签坐标系不一致。改成**先把二值掩膜贴回整幅画布，再用和左图同一个 `draw_detections()` 在绝对坐标系里画**，两个面板圆圈/标签完全一致。顺带删掉了不再使用的 `ball_detector.draw`（`draw_balls`）导入。
5. **【功能】网站开关默认关闭**：`--no-stream`（默认开、要关手动加）反过来改成 `--stream`（`store_true`，默认 `False`，需要用手机/电脑看画面时再显式加）。`run_gui()` 里 `stream_server` 的构造条件同步从 `if not args.no_stream` 改成 `if args.stream`，未开启时打印提示。

**效果/验证**（本机实际接了 RealSense D435，全部是真实硬件测得的数据，不是合成帧）：
- 修复前(当前代码路径复现)：协商到 640x480 **rgb8@15fps**，实测交付 **~12.6fps**。
- 修复后：协商到 640x480 **bgr8@30fps**，实测交付 **~25.9fps**（纯读帧）/ **~27.6fps**（读帧+ROI+detect 完整管线）。
- 用真实 CLI 入口跑 `--headless --no-mcu` 8 秒：日志确认"未开启（默认关闭）"网站提示、协商到 30fps、曝光成像锁定成功、`detect_ms_ema` 稳定在 1ms，**headless 状态行稳定在 30.0FPS**（相当于打满了这台 D435 在这个分辨率/格式下能给到的硬件上限），且真实检测到球（`ball_x=576`）。
- 无硬件的合成帧回归测试（含新增的"左右面板坐标一致性"用例）全部通过，`py_compile` 通过。

**关于"能不能到60fps"**：这颗 D435 在 640x480 分辨率、bgr8 格式下，SDK 层面能给到的最高档位就是 30fps（6/15/30三档，没有60），这是传感器/驱动的硬性限制，代码改不动。30fps 已经是这个分辨率下能拿到的上限；真要更高只能降分辨率（比如 424x240 这颗设备是有 60fps bgr8 档位的，但会牵动标定/去畸变映射/ROI绝对像素坐标/串口协议坐标含义，属于更大改动，本轮未做）。

**回滚**：`Formal_code/` 未纳入 git，需手动改回（`v1.1_beta.bgsub.bak.py` 是更早的背景差分基线，不含这轮改动，不能直接用来回滚这几条）。

---

## 2026-07-30 —— 修复：右侧调参预览面板和左图一样、不是黑白的

**动机**：用户反馈换成 BallDetector 后，主窗口右侧那块"调参预览"和左边彩色画面看起来一样，也不是预期的黑白图，没法用来调参。

**根因（真 bug）**：上一轮把右面板从旧的 absdiff 二值 mask 换成"ROI 检测预览"时，实现直接把 **彩色 ROI 子图**（`sub.copy()`）贴进右侧画布，只是少了左图的"ROI 外调暗"效果——默认 ROI=整幅画面时，左右两块贴的是同一份彩色像素、画的是同一批检测圈，肉眼看基本一样，而且从头到尾没有黑白化，跟用户预期（也是老版本的体验）不符。

**改了哪些文件**：`Formal_code/rollball_code/v1.1_beta.py`，主循环 GUI 绘制那一段。
- 右面板改成**高光二值掩膜**：ROI 内 HSV 的 V 通道按 `detector.hi_v` 阈值二值化（`cv2.inRange`，V>hi_v→白，否则→黑），这正是 `BallDetector._score()` 高光评分(specular)项实际用到的那个阈值，比"再贴一份彩色图"更贴合调参需求——钢珠高光够不够亮、`hi_v`/`min_vmax`/曝光调得对不对，一眼就能看高光有没有变成白色团块、团块位置是否和绿色检测圈对得上。仍叠加候选圆(`draw_balls`)方便对照。
- 相应更新了窗口标题打印、`PANEL_W` 注释里的面板说明文字。

**效果/验证**：合成图直接跑新逻辑确认——输出确实是纯黑白二值图（三通道相等、像素只有0/255），且和原彩色 ROI 子图不同（不再是"左右一样"）；完整集成验证脚本重跑无回归。上机验证留给用户：亮钢珠在右面板应显示为白色团块，背景黑，拖 `hi_v`/`min_vmax` 滑条应能看到白色区域变化。

**回滚**：`Formal_code/` 未纳入 git，需手动改回。

---

## 2026-07-30 —— 去掉发给下位机的坐标里的类卡尔曼平滑/续帧（改成逐帧原始值）

**动机**：用户反馈串口发的坐标好像被加了类似卡尔曼滤波的效果（拖尾/续报），这不是想要的——想要"最实时更新的数据"，滤波交给下位机做。同时确认了另一个问题（相机成像锁定是否每次都固定）：答案是**固定**——`camera_factory()`（`v1.1_beta.py`）每次开机都直接把 `rollball_config.json` 里保存的 `rs_exposure/rs_gain/rs_white_balance` 强制下发给 sensor（先关自动，`_lock_color_imaging()`），全程没有"先让自动曝光跑一段再锁定"这一步，也不是死绑第一次开机的值——而是每次都用"上一次调完按 s 存的值"，改滑条+存盘后下次开机就用新值。这条不用改代码，只是确认现状。

**根因（真实存在，非用户误解）**：上一轮接入 `BallDetector` 时用了它自带的 `pick_primary()`（EMA 平滑，新旧值按 `ema_alpha` 混合）+ `detect()` 内部的漏检续帧（本帧没找到候选但历史置信度>0时，用【速度外推的预测位置】顶替续报，score 变成置信度而非真实评分）。这两个机制叠加就是典型的"类卡尔曼"效果：位置拖尾、丢球时还继续报一个编出来的坐标。

**改了哪些文件**：仅 `Formal_code/rollball_code/v1.1_beta.py`（`code/opencv_code/ball_detector.py` 不动——它是共享代码，`code/opencv_code/v1.0.py`/`v1.1.py`/`tune_detector.py` 还依赖它现在的行为，不能改共享文件的默认语义）。
- 主循环：`primary = detector.pick_primary(cands)` 改成 `primary = cands[0] if cands else None`（不平滑，取本帧原始最高分候选）+ 紧跟一行 `detector.reset()`（每帧检测完立即清空检测器的位置/速度/置信度跟踪状态）。
- `reset()` 让下一帧进 `detect()` 时 `self._px` 恒为 `None` → 内部"漏检续帧"分支的触发条件（`self._px is not None`）永远不满足 → 没检测到就必然返回 `[]` → `primary=None` → `mcu.send_x(None)` → 立即发 `$X,NA`，不会用上一帧位置+速度编坐标。
- 相应更新了模块顶部/主循环处的注释，说明为什么不用 `pick_primary()`、`reset()` 的作用，以及副作用（`_score()` 的"空间一致性奖励"因跨帧参考被清空而拿不到，评分会略低，但不影响 Hough 找圆本身，不影响稳定检出）。

**效果/验证**（无硬件，合成帧直接调用 `BallDetector` 对比新旧两种用法）
- 位置突变测试：球从 x=80 瞬间跳到 x=160，新路径(raw+reset)第2帧读数≈159（立即跟上）；对照旧路径(`pick_primary`) 第2帧读数被拖到 103（EMA 拖尾），证实新路径确实消除了平滑。
- 漏检续帧测试：命中一帧后紧跟一帧全黑无球，新路径立即 `primary=None`；对照旧行为（不 reset）在同样丢球帧会续报一个 `score=0.225`(置信度) 的编造坐标，证实新路径消除了续帧。
- 完整集成验证脚本（模块导入/配置读写/检测管线/绘制/坐标换算）重跑全绿，无回归。

**权衡（告知用户）**：现在任何一帧没检测到就立即发 NA，不再有"短暂遮挡/闪烁时靠预测撑一下"的容错——如果后续发现下位机那边因为偶发单帧丢检抖动太多，可以再讨论要不要在协议层加低成本的"连续 N 帧都丢才报 NA"这种极简防抖（而不是恢复位置平滑）。

**回滚**：`Formal_code/` 未纳入 git，需手动改回（把 `primary = cands[0]...` + `detector.reset()` 两行换回 `primary = detector.pick_primary(cands)`，删掉 reset 调用）。

---

## 2026-07-30 —— 检测鲁棒性大改：锁定 RealSense 成像 + 用 ball_detector(形状+高光) 替代静态背景差分

**动机**：用户反馈"每次进去都不一定识别到钢珠，调了阈值也不稳；光照变一点就崩"，项目还要搬去别的环境跑，需要鲁棒性。两路只读排查确认两个独立根因（都要修）：
1. **相机成像每次不一样**：`RealSenseCamera` 从没锁过成像，彩色 sensor 的自动曝光+自动白平衡默认全开 → 每次开机/光照微变都重新自适应、像素值就变，上次调的阈值必然失配。
2. **算法脆弱**：静态背景差分（当前帧 vs 一次性拍的 `background.png` 做 absdiff）→ 任何全局光照变化让整帧都偏离背景、mask 泛白 → 检测崩。
用户确认：目标是**银色反光钢珠**（有高光点）、**接受锁定相机参数换可复现**（每到新环境调一次）。

**改了哪些文件**（仅 `Formal_code/rollball_code/v1.1_beta.py`；`code/opencv_code/ball_detector.py` 只复用不改）
- **层1 锁定成像**：`RealSenseCamera` 起流后取彩色 sensor，**先关自动曝光/自动白平衡、再设固定 exposure/gain/white_balance**（顺序同 camera_common 的 V4L2 约定）。抽成 `_lock_color_imaging()`（主起流+降级两分支都调）、`_get_color_sensor()`（first_color_sensor 退而 query_sensors）、`set_color_options()`（按 sensor 实际量程夹紧再设、越界不报错、记住最新值供重连重设）。固定值来自 cfg 的 `rs_exposure/rs_gain/rs_white_balance`，做成 Tuning 滑条**实时可调**（拖动→`cam.set_color_options`→存盘），掉线重连经 `camera_factory` 用最新值重新锁定。`open_camera` 加 `rs_opts` 参数透传，`ResilientCamera.set_color_options` 转发给底层 cam。
- **层2 换检测**：`sys.path` 加 `code/opencv_code`，复用 `ball_detector.BallDetector`（HoughCircles 找圆 + HSV V 通道高光/亮度/对比度多因素评分 + 速度预测跟踪/置信度衰减，**不依赖背景图**）。主循环 `cands = detector.detect(sub)` → `primary = detector.pick_primary(cands)`（ROI 内坐标，+`x0` 发串口）。漏检时靠预测续帧，连续性比原来逐帧选最大更好。
- **删除背景差分整套**：`detect()`、`calibrate_background/quick_background/load_background`、`BG_FILE/background.png` 依赖、`b` 键、`bg_full/bg_is_temp`、mask 面板、`odd()`，以及 cfg 里 `thresh/auto_thresh/blur_k/open_k/close_k/min_area/max_area/min_circ`。
- **Tuning 滑条改版**：ROI 四条 + 检测器(`param2/min_vmax/hi_v/min_radius/max_radius`，实时写进 detector 对象) + 成像(`rs_exposure/gain/white_balance`)。右面板从 absdiff mask 改成 **ROI 检测预览**（画候选圆+主目标，调 param2/曝光的可视反馈）。`s`/退出把 `detector.as_dict()` 整包存进 `cfg["detector"]`。
- **配置迁移**：`load_cfg` 只认 `DEFAULT_CFG` 键 → 旧 json 的背景差分字段自动忽略、新键(rs成像/detector)回退默认，不报错。
- 背景差分基线备份在 `v1.1_beta.bgsub.bak.py`（`Formal_code/` 未纳入 git）。

**协同点**：锁定曝光后钢珠高光亮度稳定 → ball_detector 里那些基于绝对 V 阈值(min_vmax/hi_v)的评分才可复现。两层互相强化：调一次、重启仍准。

**效果/验证**（本机无 RealSense，用 importlib+合成帧端到端验证，全绿）
- 模块及所有依赖导入成功；旧 `detect()`/背景标定函数确认已删除。
- 配置 roundtrip：`rs_exposure`、`detector` 参数整包(param2/min_vmax)持久化并能 `load_dict` 恢复；旧 schema(含 thresh/min_area) 平滑迁移不报错。
- 检测管线+绘制：`apply_rect_roi`→`detect`→`pick_primary`→`draw_balls`/`build_detection_overlay` 全程无异常；正检出路径实测 detect 返回候选(99,61,14,score64)、坐标换算 abs_x 正确、绘制通过。
- **相机锁定/实际检测鲁棒性未上机验证**（无 RealSense）：需上机确认关自动曝光/白平衡日志、拖曝光/param2/min_vmax 把钢珠调出来、`s` 存、**重启仍能直接识别**（可复现）、轻微改光照/换位置仍稳定。回归录像/颜色/帧率/网站。

**回滚**：`Formal_code/` 未纳入 git。回滚用 `cp v1.1_beta.bgsub.bak.py v1.1_beta.py`（恢复背景差分版）。建议尽快 `git add Formal_code/`。

---

## 2026-07-30 —— 录像时长过短/颜色红蓝互换两 bug 修复 + 录像&推流编码搬后台线程提帧率

**动机**：上机测试反馈两点 —— ①下位机 START→STOP 约 5 秒，但保存的 mp4 只播 ~1 秒；②录像里红色物体显示成蓝色（判断为通道转换错）。同时要求：在保留"保存视频 + MJPEG 网站"的前提下**尽可能提高帧率**。用户明确选择**加后台编码/IO 线程**（采集/检测/发串口主循环仍单线程），部分放宽了此前"以稳定否决多线程"的决定。

**根因（两路只读排查确认）**
- **时长过短**：`mcu_link.py::_open_writer` 用固定 `video_fps`（来自 `--fps`，默认 **60**）建 VideoWriter，但录像是主循环每帧写一帧、真实速率只有十几~三十几 fps。用 60 的头封装 ~12fps 的帧 → 时长=帧数/60 → 5s 压成 ~1s。
- **颜色互换**：`v1.1_beta.py::RealSenseCamera.read()` 丢了参考实现 `ruikang/my/v2.8.py` 里的运行时格式判断。主起流请求 bgr8，但 `640x480@60` 协商失败会落到**无格式降级分支**，驱动给 rgb8，read() 原样返回 → 全局红蓝互换（预览/网页/录像一致，只是用户在录像里才注意到）。

**改了哪些文件**
- `Formal_code/rollball_code/v1.1_beta.py`
  - `RealSenseCamera.read()`：加回运行时 guard——`if color_frame.profile.format() == self._rs.format.rgb8: cvtColor(RGB2BGR)`，bgr8 不转、rgb8 才转（修颜色）。
  - 主循环写帧调用改成 `mcu.write_video_frame(frame, measured_fps=fps_ema)`（把实测帧率喂给录像）。
- `Formal_code/rollball_code/mcu_link.py`
  - 新增 **`_VideoRecorder`**：后台写盘线程 + 有界队列（maxsize=60）。VideoWriter 由该线程**独占**创建/写/释放（OpenCV VideoWriter 非线程安全）；主线程 `write_video_frame` 只**非阻塞入队**帧引用，队列满则**丢帧+计数告警**（宁丢不阻塞主循环）；水印由 worker 在自己的 copy 上画。`_open_writer` 迁到 worker，用 **measured_fps**（夹 `[1,120]`，无效回退 video_fps）建 writer 并打印实际 fps（修时长）。
  - `McuLink`：`recording` 改看主线程标志 `_rec_active`（不再看 `writer`）；`_start_task/_stop_task/close` 适配后台线程生命周期（STOP 只入队关闭哨兵不阻塞、close 停线程 join 兜底 release）。删除旧的 `self.writer/_open_writer/_close_writer/_pending_new_recording`。
- `Formal_code/rollball_code/mjpeg_server.py`
  - 新增后台 **JPEG 编码线程**：`update_frame()` 改成只存最新帧引用 + 唤醒编码线程（非阻塞、latest-wins），`cv2.imencode` 从主循环搬到编码线程；`start/stop` 起停该线程（stop 先停编码线程再置 `_jpg=None` 让推流线程退出）。

**效果/验证**
- `py_compile` 三个文件通过。
- **无硬件端到端验证脚本全绿**（scratchpad）：
  - `_VideoRecorder`：75 帧@实测 25fps → 读回 mp4 恰为 25fps/75 帧/**3.00s**（时长与实际一致，不再压短）；猛灌 500 帧（队列仅 10）**不阻塞不崩、正确丢帧 482**。
  - `McuLink` 全状态机：START→写 40 帧(measured 20fps)→STOP → 生成 mp4 读回 **20fps**（不是 video_fps=60）/40 帧齐。
  - `MjpegServer`：`update_frame` 0.04ms 非阻塞（未在主线程 imencode）、编码线程产出合法 JPEG(FFD8)、能干净停。
- **颜色 guard 未上机验证**（本环境无 RealSense）：改动是对参考实现的直接镜像，逻辑等价；上机需实测确认红为红。

**帧率预期**：录像时 `writer.write()` 与推流 `imencode` 都已离开主循环，headless+看网页+录像时主循环≈只剩 采集+检测+发串口，应稳定在接近相机硬顶 ~40fps（改前录像会明显下掉）。留意后台若打印"写盘跟不上丢帧"告警。

**回滚**：`Formal_code/` 仍未纳入 git，无法 `git checkout`，需手动改回（撤销上述三处；`serial_link.py` 未动）。建议尽快 `git add Formal_code/`。

---

## 2026-07-30 —— 修复"发保存视频指令后没录像"的死锁 bug + ROI 从条带升级为矩形（新增左右边界滑条）

**动机**：用户反馈下位机发 TASK START 后视频没有正常保存；同时希望在现有的上/下两条 Y 边界基础上，再加两条 X 边界，并且要用滑块交互式框选，缩小感兴趣区域。

**问题 1 根因（录像不保存）**
- `mcu_link.py::write_video_frame()` 设计上要求**每帧无条件调用**，第一次调用时才会
  懒创建 `self.writer`（此前只是 START 时置了 `_pending_new_recording=True`）。
- 但 `v1.1_beta.py` 主循环里包了一层 `if mcu.recording: mcu.write_video_frame(frame)`，
  而 `mcu.recording` 的定义正是 `self.writer is not None`。于是 START 之后
  `mcu.recording` 永远是 `False`（因为 writer 还没被创建），`write_video_frame` 永远
  不会被调用，`self.writer` 也就永远建不起来——**PING/PONG、TASK/ACK 握手全部正常，
  但视频文件从来没被创建过**，跟"保存指令没生效"的现象完全吻合。
- `mcu_link.py` 顶部的类文档字符串（用法示例）当时写的就是这个错误模式，调用方是照抄的。

**改了哪些文件**
- `Formal_code/rollball_code/v1.1_beta.py`：主循环里的
  `if mcu.recording: mcu.write_video_frame(frame)` 改成无条件的
  `mcu.write_video_frame(frame)`（每帧都调，方法内部自己判断要不要真的写，开销可忽略）。
- `Formal_code/rollball_code/mcu_link.py`：
  - 类文档字符串里的用法示例同步改成无条件调用，并在 `write_video_frame()` 方法本身加了
    "踩坑"说明，防止以后又被包一层 `if mcu.recording`。

**问题 2/3（矩形 ROI + 滑块框选）**
- 原来只有 `roi_top`/`roi_bottom` 两条 Y 方向边界（滑条），裁出一条贯穿全宽的水平条带。
  新增 `roi_left`/`roi_right` 两条 X 方向边界，四条边界都是 Tuning 窗口里的滑条，
  拖动即时生效——上下左右四条线框出一个矩形 ROI，比原来的条带更具体。
- 涉及的改动点（`Formal_code/rollball_code/v1.1_beta.py`）：
  - `DEFAULT_CFG` 新增 `roi_left`/`roi_right`（语义和 `roi_top`/`roi_bottom` 对称：
    0 表示待初始化为整幅宽度，`right<=left` 视为未设置）。
  - `apply_band_roi()` 改名/扩展为 `apply_rect_roi(frame, top, bottom, left, right)`，
    同时裁剪 X/Y 两个方向，返回 `(sub, (x0, y0))`。
  - Tuning 窗口新增 `roi_left`/`roi_right` 两个滑条（`_TRACKBAR_ORDER`/`setup_trackbars`/
    `read_trackbars`/`SLIDER_HELP`），窗口高度从 380 加到 460 装下新滑条。
  - `draw_band_highlight()` 改名 `draw_roi_highlight()`，从只调暗上下改成调暗矩形 ROI
    外的全部区域，画四条黄色边界线（上/下/左/右）。
  - `draw_detections()`/`build_detection_overlay()` 加上 `x_offset` 参数——此前默认
    `x0` 恒为 0（条带裁全宽）没问题，现在 ROI 有左边界后候选/主目标的绝对坐标必须加
    `x0` 才不会画歪。
  - **重要**：`mcu.send_x()` 发的是全画面绝对像素 x，之前 `best["cx"]` 就是绝对坐标
    （因为条带不裁 X），现在 `best["cx"]` 是 ROI 内相对坐标，改成
    `mcu.send_x(int(best["cx"]) + x0 ...)`，否则加了左边界后下位机收到的 X 会整体偏小。
  - 掩膜面板 `mask_full` 嵌入偏移量、`c` 清空 ROI 按键、摄像头重连后分辨率变化的重置逻辑，
    都同步从"只处理 Y"扩到"X/Y 都处理"。
  - `run_selftest()` 同步支持 `roi_left`/`roi_right`（自检也走同一套矩形裁剪路径）。

**效果/验证**
- `py_compile` 通过两个改动文件，无语法错误。
- 旧的 `rollball_config.json` 没有 `roi_left`/`roi_right` 字段，`load_cfg()` 按现有合并逻辑
  （`if k in data`）会自动回落到 `DEFAULT_CFG` 的 `0,0`，运行时按"未设置"逻辑初始化为整幅
  宽度，不会因为老配置文件缺字段而报错或行为异常。
- 未接实体摄像头/单片机跑通 GUI，实机验证（录像是否真的生成 mp4、四条滑条框选是否符合
  预期）留给用户下次上机测试。

**回滚**：`Formal_code/` 未纳入 git 版本管理（`git status` 显示整个目录是 `??`），无法用
`git checkout` 回滚，需要手动改回。建议尽快 `git add Formal_code/` 纳入版本管理。

---

## 2026-07-29 —— 终端打印改回中文 + 恢复串口打印限流

**动机**：上一条把终端说明改成了英文，用户看不懂，要求改回中文；之前为了核对发送内容临时把 `$X` 打印改成逐帧不限流，现在核对完了要求恢复限流。

**改了哪些文件**
- `Formal_code/rollball_code/v1.1_beta.py`：`SLIDER_HELP`（10 条滑条含义）、启动时打印的"窗口/按键"说明，全部改回中文。**注意**：滑条**窗口控件本身**的名字（`roi_top`/`thresh`等）仍是英文 ASCII——这是本机 OpenCV Qt 组件缺字体、滑条名画不出中文的环境限制，改不了，只能靠终端打印的中文对照表看含义。
- `Formal_code/rollball_code/mcu_link.py`：`send_x()` 打印恢复限流 0.3s（`log_always=False, throttle_key="x", throttle_interval=0.3`）。发送本身不受影响，一直是每帧都发。

**效果/验证**：`py_compile` 通过；带窗口实跑，启动打印全中文、无异常。

---

## 2026-07-29 —— 去掉中文说明面板 + 串口每帧就发 + 只留英文滑条窗口

**动机**：中文说明面板画字太吃 CPU、拖慢串口发送；希望"每读到一帧有效数据就立刻发 X"，窗口只保留带滑块的调参窗口，说明用英文。

**改了哪些文件**
- `Formal_code/rollball_code/v1.1_beta.py`
  - **删除**了 PIL 中文说明面板整套：`build_info_panel()`、`draw_cn_lines()`、`_font()`、`_FONT_PATH`、`PANEL_REFRESH_INTERVAL`，以及 `from PIL import ...`、`from functools import lru_cache` 两个 import。
  - 主窗口从 `左图 | 掩膜 | 中文面板` 三栏改成 **`左图 | 掩膜` 两栏**（`PANEL_W = W*2`）。
  - 滑条名从编号 `01_roi_top…10_min_circ` 改成**英文参数名** `roi_top / roi_bottom / thresh / auto_thresh / blur_k / open_k / close_k / min_area / max_area / min_circ`（直接用 cfg 键名当滑条名）。启动时把每个滑条的英文含义打印到终端（`SLIDER_HELP`）。
  - 启动打印的窗口/按键说明也改成英文。
- `Formal_code/rollball_code/mcu_link.py`
  - `send_x()` **去掉了有效坐标的 60Hz 发送节流**：现在只要这一帧检测到球就立即发一次 `$X`，发送快慢完全由主循环帧率决定。丢球 `$X,NA` 仍按协议 §7（刚丢立即发、持续丢每 200ms）。终端**打印**仍限流 0.3s（只是别刷屏，不影响发送本身）。

**效果/验证**
- `py_compile` 通过；grep 确认无旧引用残留。
- 带窗口实跑：英文滑条说明正常打印，窗口只剩 图像+掩膜，串口正常发 `$X,307`（有球）/`$X,NA`（无球），无异常。
- 帧率：中文面板本来占 CPU ~70%，去掉后带窗口帧率≈headless 的 ~33fps。

**回滚**：`Formal_code/` 目前未纳入 git（`git status` 里是 `??`），没有历史可 checkout。要回滚就把上面删掉的函数/import 加回、滑条名改回编号、`send_x` 恢复 60Hz 节流。建议尽快 `git add Formal_code/` 纳入版本管理，以后回滚才方便。

---

## 2026-07-29 —— 帧率优化：串口发送慢的真凶是串口阻塞读

**动机**：用户觉得串口发 X 太慢（个位数 Hz），怀疑是识别慢。排查发现根因另有其人。

**实测定位**
- 识别本身很快（`detect()` ~4ms/帧），不是瓶颈。
- 中文说明面板画字单帧 25~40ms、占 70%+（本条之后已彻底删除，见上一条）。
- **真凶**：`code/task_code/serial_link.py` 的 `read_available()` 无数据时会 `ser.read(1)` 死等一个 timeout(0.2s)，**实测单次均值 116.8ms**。而 `mcu.poll_incoming()` 每帧调它一次 → 下位机一连上、整个视觉循环就被拖到 ~8Hz。
- 相机硬顶：RealSense 在树莓派5上纯读帧 ~43fps 是驱动/USB 层硬限，单线程 read+detect 串行 ~33fps。要冲 42 得把采集单独开线程——**用户以稳定为由否决了多线程**。

**改了哪些文件**
- `code/task_code/serial_link.py`：**新增**非阻塞方法 `read_available_nonblocking()`（`in_waiting<=0` 立即返回 `b""`）。**additive，不改旧 `read_available()`**，所以 `serial_test.py` 不受影响。
- `Formal_code/rollball_code/mcu_link.py`：`poll_incoming()` 改用 `read_available_nonblocking()`。
- `Formal_code/rollball_code/v1.1_beta.py`：
  - 面板改成限流重画 + `_font()` 加缓存（这两条随后被"删除面板"整个替代）。
  - 新增 `--headless` 模式：不开任何窗口，只跑 采集→检测→串口，比赛/部署用，Ctrl-C 退出。
  - GUI 重绘/imshow 限流到 ~15Hz（`DISPLAY_REFRESH_INTERVAL`），把 CPU 让给采集+串口。
  - 主循环包 `try/except KeyboardInterrupt` 以便 Ctrl-C 干净退出。

**效果/验证**：修串口阻塞后，带 MCU 实跑稳定 ~33fps；球在画面内时约 **30-33 次有效 $X/秒**（之前 ~8Hz）。

**回滚**：`serial_link.py` 已在 git 跟踪，`git checkout code/task_code/serial_link.py` 可回滚（会同时丢掉新方法）。Formal_code 未跟踪，回滚靠手动。

---

## 2026-07-29 —— 串口收发全部打印到终端

**动机**：怀疑下位机收不到 X，想在终端看清实际收发。

**改了哪些文件**
- `Formal_code/rollball_code/mcu_link.py`：加统一发送+打印 helper `_write()`；`_handle_line()` 打印收到的帧（含校验失败的原始行）。控制帧(PONG/ACK/PING/TASK)每条都打印；`$X` 高频帧限流打印(0.3s)避免刷屏。

**顺带查明**：`$X` 发的是**全画面像素 x（0~640）**，不是协议写的 mm(-130~+130)。用户决定**就发像素**，下位机那边按像素处理（协议文档单位注释宜同步改成"像素"，避免后续被误导）。
