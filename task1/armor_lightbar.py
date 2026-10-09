from pathlib import Path
import sys
import time

import cv2
import numpy as np


# -------------------- 1. 路径和初始参数 --------------------

VIDEO_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "data/videos/1.webm"
)

OUTPUT_DIR = Path("output/task1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 参数已基于第300帧进行实际调整（HSV与B-R交集）
HSV_LOWER = np.array([90, 50, 40], dtype=np.uint8)
HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
B_R_THRESHOLD = 15

KERNEL_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
KERNEL_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


# -------------------- 2. 保存图片的小工具 --------------------

def save_image(filename, image):
    path = OUTPUT_DIR / filename
    if not cv2.imwrite(str(path), image):
        print(f"警告：图片保存失败：{path}")

def make_tile(image, title, size=(420, 260)):
    tile_w, tile_h = size
    title_h = 28
    content_h = tile_h - title_h

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(tile_w / w, content_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    tile[title_h:title_h + new_h, :new_w] = resized

    cv2.putText(tile, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


# -------------------- 3. 颜色分割和形态学对比 --------------------

def build_masks(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    b, g, r = cv2.split(frame)
    diff = cv2.subtract(b, r)
    _, mask_diff = cv2.threshold(diff, B_R_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 最终颜色掩膜：HSV 和 B-R 差分取交集
    blue_mask = cv2.bitwise_and(mask_hsv, mask_diff)

    # 形态学对比
    open_3 = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, KERNEL_3)
    open_5 = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, KERNEL_5)
    close_3 = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, KERNEL_3)

    # 最终掩膜：使用闭运算连接可能断裂的灯条，不要用开运算破坏小目标
    final_mask = close_3.copy()

    return blue_mask, open_3, open_5, close_3, final_mask


# -------------------- 4. 根据几何特征筛选灯条 --------------------

def find_lightbars(mask):
    height, width = mask.shape
    image_area = height * width

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5: # 降低最小面积
            continue

        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect

        long_side = max(rw, rh)
        short_side = min(rw, rh)

        if short_side < 1.5:
            continue

        if long_side < max(5.0, height * 0.005): # 放宽长边限制
            continue

        aspect_ratio = long_side / max(short_side, 1e-6)
        if aspect_ratio < 1.5:
            continue

        rect_area = rw * rh
        if rect_area > image_area * 0.03:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        candidates.append({
            "contour": contour,
            "rect": rect,
            "polygon": polygon,
            "area": area,
            "aspect_ratio": aspect_ratio,
        })

    return contours, candidates


# -------------------- 5. 制作第一帧的分析材料 --------------------

def save_first_frame_analysis(frame):
    b, g, r = cv2.split(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blue_mask, open_3, open_5, close_3, final_mask = build_masks(frame)
    contours, candidates = find_lightbars(final_mask)

    contour_view = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) >= 3:
            cv2.drawContours(contour_view, [contour], -1, (0, 255, 0), 1)

    polygon_view = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        cv2.polylines(polygon_view, [polygon], True, (0, 255, 255), 1)

    detection_view = frame.copy()
    for item in candidates:
        box = cv2.boxPoints(item["rect"]).astype(np.int32)
        cv2.polylines(detection_view, [box], True, (0, 0, 255), 2)

    cv2.putText(detection_view, f"First frame candidate bars: {len(candidates)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    save_image("original.png", frame)
    save_image("channel_B.png", b)
    save_image("channel_G.png", g)
    save_image("channel_R.png", r)
    save_image("gray.png", gray)
    save_image("hsv_blue_mask.png", blue_mask)
    save_image("morph_open_3x3.png", open_3)
    save_image("morph_open_5x5.png", open_5)
    save_image("morph_close_3x3.png", close_3)
    save_image("final_mask.png", final_mask)
    save_image("contours.png", contour_view)
    save_image("polygon_approx.png", polygon_view)
    save_image("detections_first_frame.png", detection_view)

    panels = [
        (frame, "Original"), (b, "B channel"), (g, "G channel"), (r, "R channel"),
        (gray, "Gray"), (blue_mask, "Color Mask (HSV & B-R)"), (open_3, "Open 3x3"), (open_5, "Open 5x5"),
        (close_3, "Close 3x3"), (contour_view, "Contours"), (polygon_view, "Polygon approximation"), (detection_view, "Initial detection"),
    ]

    tiles = [make_tile(image, title) for image, title in panels]
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    montage = np.vstack(rows)
    save_image("diagnostics_montage.png", montage)

    print(f"第一帧候选灯条数量：{len(candidates)}")
    print("第一帧分析图片已生成。")


# -------------------- 6. 处理一帧并绘制识别结果 --------------------

def process_frame(frame, frame_index):
    start = time.perf_counter()
    _, _, _, _, mask = build_masks(frame)
    _, candidates = find_lightbars(mask)
    result = frame.copy()

    for index, item in enumerate(candidates, start=1):
        box = cv2.boxPoints(item["rect"]).astype(np.int32)
        cv2.polylines(result, [box], True, (0, 0, 255), 2)
        center = item["rect"][0]
        label_x = max(0, int(center[0]))
        label_y = max(20, int(center[1]))
        cv2.putText(result, str(index), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    elapsed_ms = (time.perf_counter() - start) * 1000
    info = f"Frame: {frame_index + 1}  Lightbars: {len(candidates)}  Time: {elapsed_ms:.2f} ms"
    cv2.putText(result, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return result, len(candidates)


# -------------------- 7. 主程序 --------------------

def main():
    if not VIDEO_PATH.is_file():
        print(f"错误：找不到视频文件：{VIDEO_PATH.resolve()}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"错误：OpenCV 无法打开视频：{VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 60:
    fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    print("=" * 55)
    print("输入视频：", VIDEO_PATH.resolve())
    print(f"视频分辨率：{width} x {height}")
    print(f"视频帧率：{fps:.2f}")
    print(f"视频总帧数：{frame_count}")
    print("=" * 55)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        print("错误：视频已打开，但第一帧读取失败。")
        sys.exit(1)

    save_first_frame_analysis(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 修改这里：输出 avi 格式，使用 XVID 编码
    output_video = OUTPUT_DIR / "annotated_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        print("错误：无法创建输出 AVI 视频。")
        sys.exit(1)

    frame_index = 0
    total_detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result, count = process_frame(frame, frame_index)
        writer.write(result)
        total_detections += count
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"处理进度：{frame_index} 帧，当前帧候选灯条数：{count}")

    cap.release()
    writer.release()

    print("=" * 55)
    print("视频处理结束。")
    print(f"实际处理帧数：{frame_index}")
    print(f"各帧候选灯条数累计：{total_detections}")
    print(f"标记视频：{output_video.resolve()}")
    print("所有分析图片位于：", OUTPUT_DIR.resolve())
    print("=" * 55)

if __name__ == "__main__":
    main()
