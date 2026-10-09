from pathlib import Path
import cv2
import numpy as np

video_path = Path("data/videos/1.webm")
out_dir = Path("output/task1")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"无法打开视频：{video_path.resolve()}")

ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    raise SystemExit("无法读取视频第一帧")

# 方法一：HSV 蓝色分割
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_hsv = cv2.inRange(
    hsv,
    np.array([90, 80, 60], dtype=np.uint8),
    np.array([135, 255, 255], dtype=np.uint8)
)

# 方法二：B-R 通道差分
b, g, r = cv2.split(frame)
diff = cv2.subtract(b, r)

masks = {
    "HSV": mask_hsv,
}

for threshold in (20, 35, 50):
    _, mask = cv2.threshold(
        diff, threshold, 255, cv2.THRESH_BINARY
    )
    masks[f"B-R threshold {threshold}"] = mask

# 方法三：同时满足 HSV 与 B-R 差分条件
_, mask_diff_20 = cv2.threshold(
    diff, 20, 255, cv2.THRESH_BINARY
)

masks["HSV AND B-R>20"] = cv2.bitwise_and(
    mask_hsv, mask_diff_20
)

# 方法四：HSV 或 B-R 满足其中一个条件
masks["HSV OR B-R>20"] = cv2.bitwise_or(
    mask_hsv, mask_diff_20
)

def make_tile(mask, title, width=600, height=370):
    resized = cv2.resize(
        mask, (width, height - 35),
        interpolation=cv2.INTER_NEAREST
    )
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    tile[35:, :] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        tile, title, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return tile

tiles = []

print("第一帧颜色分割对比")
print("-" * 65)

for name, mask in masks.items():
    filename = (
        name.lower()
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("-", "_")
        .replace("&", "and")
    )
    filename = "".join(
        c for c in filename if c.isalnum() or c == "_"
    )
    image_path = out_dir / f"mask_{filename}.png"
    cv2.imwrite(str(image_path), mask)

    white_pixels = cv2.countNonZero(mask)
    white_percent = white_pixels / mask.size * 100

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    useful_contours = sum(
        1 for c in contours if cv2.contourArea(c) >= 3
    )

    print(
        f"{name:18s} | "
        f"白色像素: {white_pixels:7d} | "
        f"占比: {white_percent:6.3f}% | "
        f"面积>=3的轮廓: {useful_contours}"
    )

    tiles.append(make_tile(mask, name))

# 2列布局，单张图比较更容易观察。
rows = []
for i in range(0, len(tiles), 2):
    pair = tiles[i:i + 2]
    if len(pair) == 1:
        pair.append(np.zeros_like(pair[0]))
    rows.append(np.hstack(pair))

comparison = np.vstack(rows)
comparison_path = out_dir / "color_method_comparison.png"

if not cv2.imwrite(str(comparison_path), comparison):
    raise SystemExit("颜色分割对比图保存失败")

print("-" * 65)
print("对比图已保存：", comparison_path.resolve())
print("各方法的单独掩膜也已保存到：", out_dir.resolve())
