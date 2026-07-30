#!/usr/bin/env python3
"""
钢珠检测调参工具 —— v2 版
环境: /home/hao/vision_env/bin/python3

画面：绿圈=检测到的钢珠，粗绿+坐标=主目标（串口发的）。
按 s 保存到 detector_config.json，q 退出。

核心逻辑（v2 新算法）：
  检测 + 跟踪融合 —— 检测到即锁，短暂丢失时按速度惯性前推，
  置信度逐步衰减而非硬 10 帧保持，更平滑、更持久。
  满置信度时 ~40 帧（~1.3s）连续无检测才丢锁。
"""

import os, sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "ready_code"))
sys.path.insert(0, _HERE)
import camera_common as cc, config

_FONT_PATH = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/arphic/ukai.ttc"]

def _font(s):
    for fp in _FONT_PATH:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, s)
            except: pass
    return ImageFont.load_default()

def box(img, lines, x, y, sz=13, color=(255,255,255), bg=(40,40,40,170)):
    if not lines: return
    f=_font(sz); lh=sz+6
    dt=ImageDraw.Draw(Image.new("RGB",(1,1)))
    mw=max(dt.textbbox((0,0),l,font=f)[2]-0 for l in lines)
    ov=img.copy()
    cv2.rectangle(ov,(x-4,y-2),(x+mw+8,y+lh*len(lines)+6),bg[:3],-1)
    cv2.addWeighted(ov,bg[3]/255.,img,1-bg[3]/255.,0,img)
    pil=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    d=ImageDraw.Draw(pil)
    for i,l in enumerate(lines): d.text((x,y+lh*i),l,font=f,fill=color)
    cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR,dst=img)

def slider(w,n,v,m): cv2.createTrackbar(n,w,int(v),m,lambda _:None)

def main():
    cap=cc.open_camera()
    if cap is None: return
    cal=cc.load_calibration()
    maps=None
    if cal: maps=cc.build_undistort_maps(cal["camera_matrix"],cal["dist_coeffs"],cal["image_size"],alpha=0.0)

    cfg=config.load()
    from ball_detector import BallDetector
    det=BallDetector().load_dict(cfg.get("detector",{}))

    WIN="Tune | s=save q=quit"
    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,800,750)
    cv2.waitKey(1)

    slider(WIN,"1.Circle strictness",det.param2,50)
    slider(WIN,"2.Min Vmax(高光亮度)",det.min_vmax,255)
    slider(WIN,"3.EMA smooth(x100)", int(det.ema_alpha*100), 100)

    print("""参数说明:
  1.圆心严格度  - (22适中, 配合v2评分防误检, 不需要像旧版拉到32)
  2.最低高光    - 圆内最亮像素阈值(160, 降低可抓更远但可能误检)
  3.EMA平滑     - 输出坐标平滑度(0.3, 越小越平滑/越滞后)
  4.Track信心   - 算法内部跟踪置信度(非滑块, 实时显示, >0就持续报位置)
按 s 保存, q 退出
v2 核心: 检测+跟踪融合, 速度惯性前推+置信度衰减替代硬保持""")

    lm=None
    while True:
        ok,fr=cap.read()
        if not ok: continue
        if maps is not None:
            ms=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if ms!=lm: maps=cc.build_undistort_maps(cal["camera_matrix"],cal["dist_coeffs"],ms,alpha=0.0); lm=ms
            fr=cv2.remap(fr,maps[0],maps[1],cv2.INTER_LINEAR)

        H=fr.shape[0]
        det.param2    =max(1,cv2.getTrackbarPos("1.Circle strictness",WIN))
        det.min_vmax  =max(0,cv2.getTrackbarPos("2.Min Vmax(高光亮度)",WIN))
        det.ema_alpha =max(0.01,cv2.getTrackbarPos("3.EMA smooth(x100)",WIN)/100.)

        cands=det.detect(fr)
        prim=det.pick_primary(cands)
        from ball_detector import draw
        draw(fr,cands,prim)

        if prim is not None:
            conf_pct = int(det._confidence * 100)
            vel_info = ""
            if abs(det._vx) > 0.5 or abs(det._vy) > 0.5:
                vel_info = f" v=({det._vx:.0f},{det._vy:.0f})"
            st=(f"钢珠({prim[0]},{prim[1]}) r={prim[2]} | "
                f"严格度={det.param2} 高光>{det.min_vmax} "
                f"EMA={det.ema_alpha:.2f} | "
                f"信心={conf_pct}%{vel_info}")
        else:
            st=(f"无钢珠 | 严格度={det.param2} 高光>{det.min_vmax}")
        box(fr,[st],5,H-28,sz=12,color=(0,255,255),bg=(30,30,30,180))

        cv2.imshow(WIN,fr)
        k=cv2.waitKey(1)&0xFF
        if k in (ord('q'),27): break
        elif k==ord('s'):
            cfg["detector"]=det.as_dict(); cfg["undistort"]=True
            config.save(cfg) and print(
                f"已保存: p2={det.param2} vmax={det.min_vmax} ema={det.ema_alpha:.2f}"
                f" conf_inc={det.conf_inc} conf_dec={det.conf_dec}")

    cap.release(); cv2.destroyAllWindows()
if __name__=="__main__": main()
