import cv2
import numpy as np
from pathlib import Path
from pupil_apriltags import Detector

TAG_SIZE = 0.1
img = cv2.imread("output/task2/test_tag.jpg")

if img is None:
    print("错误：找不到图片！请确认 test_tag.jpg 已放入 output/task2/ 目录。")
    exit()

# 关键修正：如果图片太大，自动缩小到最大边长为 800 像素
h, w = img.shape[:2]
max_dim = 800
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    print(f"图片已自动缩放至 {img.shape[1]}x{img.shape[0]} 像素，以提升检测成功率。")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 调整检测器参数，提高鲁棒性
detector = Detector(
    families='tag36h11', nthreads=4,
    quad_decimate=2.0, quad_sigma=1.0,
    refine_edges=1, decode_sharpening=0.25
)

# 注意：因为图片被缩小了，所以内参（cx, cy）也要根据新的尺寸来调整
fx, fy = 600.0, 600.0
cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
dist = np.zeros(5)

tags = detector.detect(gray, estimate_tag_pose=True, camera_params=(fx, fy, cx, cy), tag_size=TAG_SIZE)
axis_pts = np.array([[0,0,0], [TAG_SIZE*0.5,0,0], [0,TAG_SIZE*0.5,0], [0,0,TAG_SIZE*0.5]], dtype=np.float32)

if len(tags) == 0:
    print("依然没检测到！请尝试把照片裁剪一下，只保留黑白方块部分，再重新保存为 test_tag.jpg。")
else:
    for tag in tags:
        if tag.tag_id != 0: continue
        # 画角点
        for i in range(4):
            cv2.line(img, tuple(tag.corners[i].astype(int)), tuple(tag.corners[(i+1)%4].astype(int)), (0,255,0), 2)
        # 画坐标轴
        if tag.pose_R is not None:
            imgpts, _ = cv2.projectPoints(axis_pts, tag.pose_R, tag.pose_t, mtx, dist)
            imgpts = imgpts.astype(int)
            o = tuple(imgpts[0].ravel())
            cv2.line(img, o, tuple(imgpts[1].ravel()), (0,0,255), 3) # X红
            cv2.line(img, o, tuple(imgpts[2].ravel()), (0,255,0), 3) # Y绿
            cv2.line(img, o, tuple(imgpts[3].ravel()), (255,0,0), 3) # Z蓝
            t = tag.pose_t.flatten()
            cv2.putText(img, f"Dist: {np.linalg.norm(t):.3f} m", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            print(f"🎉 成功！距离: {np.linalg.norm(t):.4f} 米")
            print(f"旋转矩阵 R:\n{tag.pose_R}")
    
    cv2.imwrite("output/task2/apriltag_screenshot.png", img)
    print("结果图片已保存至 output/task2/apriltag_screenshot.png")
