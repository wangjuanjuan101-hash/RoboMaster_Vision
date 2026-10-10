import cv2
import numpy as np
from pathlib import Path
import sys

# ================= 配置区 =================
TAG_SIZE = 0.1  # AprilTag 黑色外框边长（单位：米，如果你量的是95mm就写0.095）
VIDEO_SOURCE = "data/videos/apriltag_test.mp4"  # 输入视频
OUTPUT_VIDEO = "output/task2/apriltag_demo.mp4" # 输出带有坐标轴的视频
CALIB_FILE = Path("output/task2/camera_params.npz")
# ==========================================

def main():
    # 1. 尝试加载标定结果。若无，则使用估算内参，确保程序绝对能跑起来
    if CALIB_FILE.exists():
        data = np.load(CALIB_FILE)
        mtx, dist = data['mtx'], data['dist']
        fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]
        print("成功加载标定内参。")
    else:
        fx, fy, cx, cy = 600.0, 600.0, 320.0, 240.0
        mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist = np.zeros(5)
        print("警告：未找到标定文件，使用估算内参运行。")

    # 2. 初始化 AprilTag 检测器
    try:
        from pupil_apriltags import Detector
    except ImportError:
        print("错误：未安装 pupil_apriltags。")
        return

    at_detector = Detector(
        families='tag36h11', nthreads=4, quad_decimate=1.0,
        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0
    )

    # 3. 打开视频
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {VIDEO_SOURCE}。")
        return

    # 获取输入视频的尺寸和帧率，用于初始化视频写入器
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 60:
        fps = 30.0

    # 初始化视频写入器 (使用 mp4v 编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        print("错误：无法创建输出视频，请检查 output/task2 目录是否存在及权限。")
        return

    print(f"开始处理视频：{VIDEO_SOURCE}")
    print(f"结果将保存至：{OUTPUT_VIDEO}")
    print("按 [q] 键可提前退出。")

    axis_length = TAG_SIZE * 0.5
    axis_points = np.array([
        [0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]
    ], dtype=np.float32)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tags = at_detector.detect(gray, estimate_tag_pose=True, 
                                  camera_params=(fx, fy, cx, cy), tag_size=TAG_SIZE)

        display = frame.copy()

        for tag in tags:
            if tag.tag_id != 0: continue

            center = tuple(int(c) for c in tag.center)
            cv2.circle(display, center, 5, (0, 0, 255), -1)
            
            corners = tag.corners.astype(int)
            for i in range(4):
                cv2.line(display, tuple(corners[i]), tuple(corners[(i+1)%4]), (0, 255, 0), 2)
                cv2.circle(display, tuple(corners[i]), 3, (0, 255, 255), -1)

            if tag.pose_R is not None and tag.pose_t is not None:
                imgpts, _ = cv2.projectPoints(axis_points, tag.pose_R, tag.pose_t, mtx, dist)
                imgpts = imgpts.astype(int)
                origin = tuple(imgpts[0].ravel())
                cv2.line(display, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X 红
                cv2.line(display, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y 绿
                cv2.line(display, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z 蓝

                t = tag.pose_t.flatten()
                distance = np.linalg.norm(t)
                cv2.putText(display, f"ID: {tag.tag_id}  Dist: {distance:.3f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 终端打印（只在每30帧打印一次，防止刷屏太快）
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count} | 距离: {distance:.4f} 米 | t: x={t[0]:.4f}, y={t[1]:.4f}, z={t[2]:.4f}")

        # 写入输出视频！
        writer.write(display)
        frame_count += 1

        # 实时显示（电脑屏幕上也会弹出来，可以看进度）
        cv2.imshow("AprilTag Pose", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"处理完成！共处理 {frame_count} 帧。")
    print(f"演示视频已保存至：{OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
