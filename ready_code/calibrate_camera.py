#!/usr/bin/env python3
"""
USB摄像头一键标定 - 按键采集 + 自动解算
环境: /home/hao/vision_env/bin/python3
用法: /home/hao/vision_env/bin/python3 calibrate_camera.py
注意: 需要能看到画面窗口，请在有显示器/桌面的会话里运行（不要用纯 SSH 无 X 转发）。

做什么:
  1. 用和 v1.1.py 完全相同的固定曝光/增益/白平衡打开摄像头(保证标定环境=实际
     使用环境，这样标定出来的畸变参数才准)。
  2. 实时预览并自动识别 9x6 棋盘格，识别到就把角点画成彩色连线。
  3. 按【空格】拍摄一张(只有识别到棋盘时才会真正保存)，画面左上角显示已采集张数。
  4. 采集够了按【回车】，脚本自动解算相机内参和畸变系数，打印重投影误差，
     并把结果保存到 camera_calib.npz（供 v1.2.py 去畸变使用）。
  5. 解算完成后弹出「原始 vs 去畸变」对比图，按任意键退出。

拍摄建议(直接影响标定质量):
  - 采集 15~20 张，让棋盘出现在画面的不同位置(上下左右中/四个角)。
  - 每张换一个角度：正对、左倾、右倾、上仰、下俯、远近都来一点。
  - 棋盘要完整、清晰、平整(贴硬板上别弯)，不要反光、不要运动模糊。

按键:
  空格      : 拍摄当前帧(需已识别到棋盘)
  z         : 删除最近一张采集
  回车      : 结束采集并自动标定
  q / ESC   : 放弃退出(不标定)
"""

import os
import datetime

import cv2
import numpy as np

import camera_common as cc

SHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_shots")
MIN_SHOTS = 8  # 少于这个数不允许标定

# findChessboardCorners 的标志：自适应阈值 + 归一化 + 快速预筛(实时预览用)
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
    + cv2.CALIB_CB_FAST_CHECK
)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def make_object_points():
    """生成棋盘在物理世界里的角点坐标(z=0 平面)，单位毫米。"""
    cols, rows = cc.CHESSBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= cc.SQUARE_SIZE_MM
    return objp


def run_calibration(objpoints, imgpoints, image_size):
    """解算并保存标定结果，返回 (camera_matrix, dist_coeffs, rms)。"""
    print(f"\n开始标定，共 {len(objpoints)} 张有效样本 ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # 逐张重投影误差，帮助判断有没有拍坏的样本
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_view.append(err)

    print("\n===== 标定结果 =====")
    print(f"整体重投影误差 RMS = {rms:.4f} 像素  (越小越好，< 0.5 很好，< 1.0 可用)")
    print("相机内参矩阵 camera_matrix =")
    print(camera_matrix)
    print("畸变系数 dist_coeffs (k1 k2 p1 p2 k3) =")
    print(dist_coeffs.ravel())
    worst = int(np.argmax(per_view))
    print(f"\n单张重投影误差(最差的是第 {worst} 张 = {per_view[worst]:.4f}):")
    print("  " + "  ".join(f"{e:.3f}" for e in per_view))

    cc.save_calibration(camera_matrix, dist_coeffs, image_size, rms)
    print(f"\n✅ 已保存标定结果到 {cc.CALIB_FILE}")
    return camera_matrix, dist_coeffs, rms


def show_comparison(frame, camera_matrix, dist_coeffs, image_size):
    """并排显示 原始 vs 去畸变，直观检查效果。"""
    map1, map2, _ = cc.build_undistort_maps(camera_matrix, dist_coeffs, image_size, alpha=0.0)
    undist = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    cv2.putText(frame, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(undist, "UNDISTORTED", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    combo = np.hstack([frame, undist])
    win = "Compare (any key to exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, combo.shape[1], combo.shape[0])
    cv2.imshow(win, combo)
    out = os.path.join(SHOTS_DIR, "_compare.png")
    cv2.imwrite(out, combo)
    print(f"对比图已保存到 {out}，按任意键退出。")
    cv2.waitKey(0)


def main():
    os.makedirs(SHOTS_DIR, exist_ok=True)

    cap = cc.open_camera()
    if cap is None:
        print("❌ 无法打开摄像头 /dev/video0")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = (actual_w, actual_h)
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} (固定曝光/增益/白平衡)")
    print(f"   棋盘: {cc.CHESSBOARD_SIZE[0]}x{cc.CHESSBOARD_SIZE[1]} 内角点, 方格 {cc.SQUARE_SIZE_MM}mm")
    print("   [空格]拍摄  [z]删除上一张  [回车]结束并标定  [q/ESC]放弃")

    objp = make_object_points()
    objpoints, imgpoints, saved_files = [], [], []

    win = "Calibrate Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, cc.CHESSBOARD_SIZE, FIND_FLAGS)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, cc.CHESSBOARD_SIZE, corners, found)
            status, color = "BOARD OK - press SPACE", (0, 255, 0)
        else:
            status, color = "no board", (0, 0, 255)

        cv2.putText(view, f"shots: {len(objpoints)}  ({status})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(view, "SPACE=shoot  z=undo  ENTER=calibrate  q=quit", (10, actual_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            print("已放弃，未标定。")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord(" "):
            if not found:
                print("  当前帧没识别到棋盘，未采集。")
                continue
            # 亚像素精修，提高标定精度
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            objpoints.append(objp.copy())
            imgpoints.append(refined)
            ts = datetime.datetime.now().strftime("%H%M%S_%f")[:-3]
            fn = os.path.join(SHOTS_DIR, f"shot_{len(objpoints):02d}_{ts}.png")
            cv2.imwrite(fn, frame)
            saved_files.append(fn)
            print(f"  ✅ 采集第 {len(objpoints)} 张 -> {os.path.basename(fn)}")

        elif key == ord("z"):
            if objpoints:
                objpoints.pop()
                imgpoints.pop()
                old = saved_files.pop()
                try:
                    os.remove(old)
                except OSError:
                    pass
                print(f"  已删除上一张，剩 {len(objpoints)} 张。")

        elif key in (13, 10):  # 回车
            if len(objpoints) < MIN_SHOTS:
                print(f"  至少需要 {MIN_SHOTS} 张才能标定，当前 {len(objpoints)} 张，继续采集。")
                continue
            break

    cap.release()

    if len(objpoints) < MIN_SHOTS:
        cv2.destroyAllWindows()
        print("样本不足，未标定。")
        return

    camera_matrix, dist_coeffs, rms = run_calibration(objpoints, imgpoints, image_size)

    # 用最后一帧做前后对比
    last = cv2.imread(saved_files[-1]) if saved_files else None
    if last is not None:
        show_comparison(last, camera_matrix, dist_coeffs, image_size)
    cv2.destroyAllWindows()
    print("完成。之后运行 v1.2.py，按 u 键即可切换 原始/去畸变 画面。")


if __name__ == "__main__":
    main()
