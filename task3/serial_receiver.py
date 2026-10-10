import serial
import sys

# 把这里换成你 socat 输出的第二个端口，比如 /dev/pts/3
SERIAL_PORT = '/dev/pts/2'
BAUDRATE = 115200

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except Exception as e:
        print(f"串口打开失败: {e}")
        return

    print(f"串口助手已启动，正在监听 {SERIAL_PORT}...")
    print("-" * 60)
    
    log_file = open("output/task3/serial_log.txt", "w", encoding="utf-8")
    
    while True:
        try:
            line = ser.readline().decode('ascii').strip()
            if line:
                print(f"收到: {line}")
                log_file.write(line + "\n")
                log_file.flush()
        except KeyboardInterrupt:
            print("\n接收结束。")
            break

    log_file.close()

if __name__ == "__main__":
    main()
