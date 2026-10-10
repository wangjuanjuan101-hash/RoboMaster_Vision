import serial
import time
import struct
import cv2
import numpy as np
from pathlib import Path

# 把这里换成你 socat 输出的第一个端口，比如 /dev/pts/2
SERIAL_PORT = '/dev/pts/1'
BAUDRATE = 115200

def calculate_checksum(data_str):
    checksum = 0
    for char in data_str:
        checksum ^= ord(char)
    return f"{checksum:02X}"

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except Exception as e:
        print(f"串口打开失败: {e}")
        return

    seq = 0
    start_time = time.time()
    
    print(f"正在通过 {SERIAL_PORT} 持续发送数据 (按 Ctrl+C 停止)...")
    
    # 模拟任务二的位姿结果
    # 这里我们硬编码一段有效位姿和一段无效位姿，演示“目标出现 -> 消失 -> 重现”的状态切换
    while True:
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # 模拟状态切换：前 3 秒有效，中间 2 秒无效，再 3 秒有效
        cycle = (time.time() - start_time) % 8
        if 3 <= cycle < 5:
            valid = 0
            tag_id = -1
            x_mm, y_mm, z_mm = 0.0, 0.0, 0.0
            rx, ry, rz = 0.0, 0.0, 0.0
        else:
            valid = 1
            tag_id = 0
            # 模拟位姿数据（单位：毫米，旋转向量：弧度）
            x_mm, y_mm, z_mm = 100.0, -50.0, 800.0
            rx, ry, rz = 0.000000, 0.000000, 0.000000

        # 构造 CV1 协议报文
        # $CV1,seq,t_ms,valid,id,x_mm,y_mm,z_mm,rx,ry,rz*HH
        payload = f"$CV1,{seq},{elapsed_ms},{valid},{tag_id},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},{rx:.6f},{ry:.6f},{rz:.6f}"
        checksum = calculate_checksum(payload)
        packet = f"{payload}*{checksum}\r\n"
        
        ser.write(packet.encode('ascii'))
        print(f"发送: {packet.strip()}")
        
        seq = (seq + 1) % 4294967296
        time.sleep(0.1) # 10Hz 发送频率

if __name__ == "__main__":
    main()
