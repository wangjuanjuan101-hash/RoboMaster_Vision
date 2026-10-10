import cv2
import numpy as np
from pathlib import Path
import glob

CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 14.0 # 你的实际方格边长（毫米）
CALIB_DIR = Path("data/calibration")
OUTPUT_FILE = "output/task2/camera_params.npz"

def main():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []
    images = glob.glob(str(CALIB_DIR / "*.jpg")) + glob.glob(str(CALIB_DIR / "*.png"))
    
    if len(images) < 10:
        print(f"错误：只找到 {len(images)} 张图片，至少要 10 张。")
        return

    print(f"找到 {len(images)} 张图片，开始标定...")
    gray = None
    success_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            success_count += 1
        else:
            print(f"未能检测到棋盘格: {fname}")

    if success_count < 8:
        print("有效标定图片太少，标定失败。请多拍几张清晰图片。")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)

    np.savez(OUTPUT_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, image_size=gray.shape[::-1], error=mean_error)

    print("\n" + "="*50)
    print("标定完成！")
    print(f"相机内参矩阵:\n{mtx}")
    print(f"畸变系数:\n{dist}")
    print(f"平均重投影误差: {mean_error:.4f} 像素")
    print(f"结果已保存至: {OUTPUT_FILE}")
    print("="*50)

if __name__ == "__main__":
    main()
