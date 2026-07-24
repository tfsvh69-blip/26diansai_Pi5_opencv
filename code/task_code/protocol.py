#!/usr/bin/env python3
"""
上位机(树莓派) → 下位机(单片机) 通信格式定义 —— 唯一事实来源
环境: /home/hao/vision_env/bin/python3

采用 NMEA 风格的【带校验的 ASCII 帧】：可读(串口助手直接看)、自同步($开头\\n结尾)、
带 XOR 校验(单字节翻转能查出)，单片机用 sscanf/手写状态机都好解析。

======================== 帧格式 ========================
    $BALL,<found>,<x>,<y>,<n>*<CHK>\\r\\n

  $            帧起始
  BALL         报文类型(固定)
  <found>      1=检测到球，0=没检测到
  <x> <y>      主目标球心像素坐标(整数)；found=0 时为 0,0
               坐标系：去畸变后 640x480 画面，原点左上角，x 向右、y 向下，
                       范围 x∈[0,639]、y∈[0,479]。主目标=面积最大的那个球。
  <n>          本帧检测到的球总数(整数)
  *            校验分隔符
  <CHK>        校验：$ 与 * 之间所有字符(不含$、不含*)逐字节 XOR，两位大写十六进制
  \\r\\n        帧结束(回车换行)

  例：  $BALL,1,321,240,3*0A\\r\\n     ← 检测到，主目标(321,240)，共3个球
        $BALL,0,0,0,0*XX\\r\\n         ← 没检测到球
=========================================================

下位机解析建议：
  1. 按 '$' 找帧头，'\\n' 找帧尾，取中间一整行；
  2. 从行里 '*' 处分成 payload 和 CHK 两段；
  3. 对 payload 里 "BALL,...,n"(即 $ 之后、* 之前)逐字节 XOR，与 CHK(十六进制)比对；
  4. 校验过了再 sscanf(payload, "BALL,%d,%d,%d,%d", &found,&x,&y,&n)。
  校验不过就整帧丢弃(避免用到被干扰的坏数据)。
"""

MSG_ID = "BALL"


def xor_checksum(s: bytes) -> int:
    """逐字节 XOR。"""
    c = 0
    for b in s:
        c ^= b
    return c


def build_ball_frame(found: bool, x: int, y: int, n: int) -> bytes:
    """
    组一帧待发送的字节串（含 \\r\\n）。
    found=False 时 x,y 会被强制写 0,0，与协议一致。
    """
    if not found:
        x, y = 0, 0
    payload = f"{MSG_ID},{1 if found else 0},{int(x)},{int(y)},{int(n)}"
    chk = xor_checksum(payload.encode("ascii"))
    return f"${payload}*{chk:02X}\r\n".encode("ascii")


if __name__ == "__main__":
    # 打印几个样例帧，方便下位机对照
    for args in [(True, 321, 240, 3), (True, 320, 240, 1), (False, 0, 0, 0)]:
        print(build_ball_frame(*args).decode().strip())
