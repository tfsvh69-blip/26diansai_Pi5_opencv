#!/usr/bin/env python3
"""
串口接收监视 - 实时打印下位机(单片机)发来的数据，支持热插拔
环境: /home/hao/vision_env/bin/python3（需 pyserial，已装）
用法:
    /home/hao/vision_env/bin/python3 code/task_code/serial_test.py
    # 常用参数:
    #   --any                 找不到固定口时放宽到任意 CH340/ttyUSB（默认只认固定物理口）
    #   --port /dev/ttyUSB0   手动指定串口（覆盖固定口）
    #   --baud 115200         波特率（默认 115200）
    #   --hex                 每行末尾附十六进制
    #   --raw                 不按行拆分，收到多少原样打印

热插拔行为（复用 serial_link.py）：
  - 启动时若没插设备，会【循环等待】，插上自动连接；
  - 运行中把设备拔掉，脚本不退出，提示“已断开”，插回同一个口自动【重连】继续。
  默认只认固定物理 USB 口（serial_link.PREFERRED_BYPATH），换口就改那个常量或用 --port。

按行输出：字符数据以 \\n(或 \\r\\n) 结尾时一行行干净打印，带接收时间戳。Ctrl-C 退出。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_link import SerialLink, Disconnected  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="手动指定串口（默认认固定物理口）")
    ap.add_argument("--baud", type=int, default=115200, help="波特率（默认 115200）")
    ap.add_argument("--any", action="store_true", help="放宽到任意 CH340/ttyUSB")
    ap.add_argument("--hex", action="store_true", help="每行附十六进制")
    ap.add_argument("--raw", action="store_true", help="不按行拆分，原样打印")
    args = ap.parse_args()

    link = SerialLink(port=args.port, baud=args.baud, any_ok=args.any)

    total_bytes = 0
    total_lines = 0
    t_start = time.time()
    buf = bytearray()

    def emit_line(raw_line):
        nonlocal total_lines
        total_lines += 1
        text = raw_line.decode("utf-8", errors="replace").rstrip("\r")
        out = f"[{time.strftime('%H:%M:%S')}] {text}"
        if args.hex:
            out += "   | hex: " + raw_line.rstrip(b"\r\n").hex(" ")
        print(out, flush=True)

    print("串口监视启动（Ctrl-C 退出）")
    try:
        while True:
            link.wait_and_open()          # 没设备就阻塞等待，插上自动连
            buf.clear()
            try:
                while True:
                    data = link.read_available()   # 设备被拔会抛 Disconnected
                    if not data:
                        continue
                    total_bytes += len(data)
                    if args.raw:
                        print(f"[{time.strftime('%H:%M:%S')}] "
                              f"{data.decode('utf-8', errors='replace')!r}", flush=True)
                    else:
                        buf += data
                        while b"\n" in buf:
                            idx = buf.index(b"\n")
                            emit_line(bytes(buf[:idx]))
                            del buf[:idx + 1]
            except Disconnected:
                if not args.raw and buf:      # 把断开前没收全的行也打出来
                    emit_line(bytes(buf))
                print(f"⚠️ [{time.strftime('%H:%M:%S')}] 设备已断开，等待重新插入…",
                      flush=True)
                # 回到外层 while，wait_and_open() 会一直等到重新插上
    except KeyboardInterrupt:
        pass
    finally:
        link.close()
        dur = time.time() - t_start
        rate = total_bytes / dur if dur > 0 else 0
        print(f"\n已退出。共收 {total_bytes} 字节 / {total_lines} 行，"
              f"历时 {dur:.1f}s，平均 {rate:.0f} B/s。")


if __name__ == "__main__":
    main()
