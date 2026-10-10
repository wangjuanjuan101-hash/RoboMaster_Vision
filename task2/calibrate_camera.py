import cv2
import numpy as np
import os
from pathlib import Path

# ================= 配置区 =================
CHESSBOARD_SIZE = (9, 6)     # 内角点数量（宽, 高）
SQUARE_SIZE = 14.0           # 单个方格的实际边长（改成你量出来的毫米数，1.4cm = 14.0mm）
CALIB_DIR = Path("data/calibration")
CALIB_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = "output/task2/camera_params.npz"
# ==========================================

def main():
    # 1. 准备棋盘格的三维坐标 (Z=0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] # 3D点
    imgpoints = [] # 2D点

    # 2. 采集图片
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查 VMware 是否已连接摄像头。")
        return

    print("=" * 50)
    print("相机标定图像采集")
    print("操作说明：")
    print("  按 [空格] 保存当前棋盘格图片")
    print("  按 [q] 键退出采集并开始标定")
    print("  建议采集 15~25 张，覆盖画面中心和边缘，改变距离和倾斜角度")
    print("=" * 50)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, ret_corners)
            cv2.putText(display, "Chessboard DETECTED! Press SPACE to save", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display, f"Saved: {count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1)
        if key == ord(' ') and ret_corners:
            # 亚像素级角点优化
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            img_path = CALIB_DIR / f"calib_{count:02d}.png"
            cv2.imwrite(str(img_path), frame)
            count += 1
            print(f"已保存第 {count} 张图片: {img_path}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print(f"错误：只采集了 {len(objpoints)} 张图片，至少需要 10 张才能标定。")
        return

    # 3. 执行相机标定
    print("\n正在计算相机内参和畸变系数...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # 4. 计算重投影误差
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    # 5. 保存结果
    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs,
             image_size=gray.shape[::-1], error=mean_error)

    print("\n" + "=" * 50)
    print("标定完成！")
    print(f"相机内参矩阵 (mtx):\n{mtx}")
    print(f"畸变系数 (dist):\n{dist}")
    print(f"分辨率: {gray.shape[::-1]}")
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("=" * 50)

if __name__ == "__main__":
    main()
