#!/usr/bin/env python3
"""
视觉(树莓派) <-> 单片机 UART0 通信格式定义 —— 唯一事实来源
环境: /home/hao/vision_env/bin/python3

对应文档: Formal_code/通信协议/单片机树莓派通信协议.md

======================== 通用帧格式 ========================
    $TYPE,DATA...*CHK\\r\\n

  $      帧起始
  TYPE   消息类型
  CHK    $ 与 * 之间所有 ASCII 字节逐字节 XOR，两位大写十六进制
  校验错误、字段错误、数据越界时整帧丢弃
==============================================================

MCU -> 视觉（本模块负责解析）:
    $PING,<id>*<CHK>\\r\\n
    $TASK,<run_id>,<task_id>,START*<CHK>\\r\\n
    $TASK,<run_id>,<task_id>,STOP*<CHK>\\r\\n

视觉 -> MCU（本模块负责组帧）:
    $PONG,<id>*<CHK>\\r\\n
    $ACK,<run_id>,<task_id>,START*<CHK>\\r\\n
    $ACK,<run_id>,<task_id>,STOP*<CHK>\\r\\n
    $X,<x_mm>*<CHK>\\r\\n
    $X,NA*<CHK>\\r\\n

注：本项目当前这一版 $X 帧发的是【全画面绝对像素 x 坐标】，不是协议定义的 mm
（摆杆两端像素->毫米的标定还没做，见 v1.1_beta.py 里 send_x() 调用处的说明）。
"""

INBOUND_TYPES = ("PING", "TASK")  # MCU -> 视觉，parse_frame 只需要认这两种


def xor_checksum(s: bytes) -> int:
    """逐字节 XOR。"""
    c = 0
    for b in s:
        c ^= b
    return c


def _build_frame(payload: str) -> bytes:
    chk = xor_checksum(payload.encode("ascii"))
    return f"${payload}*{chk:02X}\r\n".encode("ascii")


def build_pong_frame(ping_id) -> bytes:
    return _build_frame(f"PONG,{ping_id}")


def build_ack_frame(run_id, task_id, phase: str) -> bytes:
    """phase: "START" 或 "STOP"。"""
    return _build_frame(f"ACK,{run_id},{task_id},{phase}")


def build_x_frame(x_val: int) -> bytes:
    return _build_frame(f"X,{int(x_val)}")


def build_x_na_frame() -> bytes:
    return _build_frame("X,NA")


def parse_frame(line: bytes):
    """
    解析一条 MCU -> 视觉的帧（不含末尾 \\r\\n，serial 按 \\n 切出来的行可能还带一个
    尾部 \\r，这里会自己 strip 掉）。
    成功返回 (TYPE, [field, ...])，例如 ("TASK", ["8", "3", "START"])。
    格式不对/校验不过统一返回 None（协议要求：直接丢弃，不抛异常）。
    """
    try:
        text = line.decode("ascii").strip("\r\n").strip()
    except UnicodeDecodeError:
        return None
    if not text.startswith("$"):
        return None
    star = text.rfind("*")
    if star < 0:
        return None
    payload = text[1:star]
    chk_str = text[star + 1:]
    if len(chk_str) != 2:
        return None
    try:
        chk_recv = int(chk_str, 16)
    except ValueError:
        return None
    if xor_checksum(payload.encode("ascii")) != chk_recv:
        return None
    parts = payload.split(",")
    if not parts or not parts[0]:
        return None
    msg_type, fields = parts[0], parts[1:]
    if msg_type not in INBOUND_TYPES:
        return None
    return msg_type, fields


if __name__ == "__main__":
    # 打印样例帧，方便和单片机那边核对
    print(build_pong_frame(17).decode().strip())
    print(build_ack_frame(8, 3, "START").decode().strip())
    print(build_ack_frame(8, 3, "STOP").decode().strip())
    print(build_x_frame(-42).decode().strip())
    print(build_x_frame(0).decode().strip())
    print(build_x_na_frame().decode().strip())

    # 用 xor_checksum 现算校验位，构造几条"模拟 MCU 发来的帧"验证 parse_frame 往返正确
    ping_payload = "PING,17"
    ping_frame = f"${ping_payload}*{xor_checksum(ping_payload.encode()):02X}\r\n".encode()
    task_payload = "TASK,8,3,START"
    task_frame = f"${task_payload}*{xor_checksum(task_payload.encode()):02X}\r\n".encode()
    print(ping_frame.strip(), "->", parse_frame(ping_frame))
    print(task_frame.strip(), "->", parse_frame(task_frame))
    bad_frame = task_frame.replace(b"*", b"X*", 1)  # 破坏 payload，校验应当失败
    print(bad_frame.strip(), "-> (期望 None)", parse_frame(bad_frame))
