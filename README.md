# RoboMaster 视觉招新考核 - 

## 任务一：装甲板灯条识别

## 1. 环境依赖
- 操作系统：Ubuntu 22.04 LTS (VMware 虚拟机)
- 编程语言：Python 3.10.12
- 核心工具：OpenCV 4.12.0.88，NumPy 2.2.6
- 依赖安装：`pip install -r requirements.txt`

## 2. 运行方式
确保虚拟环境已激活，然后运行以下命令：

    source .venv/bin/activate
    python task1/armor_lightbar.py data/videos/1.webm

## 3. 演示视频下载
通过网盘分享的文件：final_video_30fps.mp4

链接：[点击下载视频](https://pan.baidu.com/s/1HXKus8wrZMM3jy_rX1IrWA?pwd=7cxj)

提取码：7cxj

---

## 任务二：相机标定与 AprilTag

### 1. 标定与硬件变通说明
由于 VMware 虚拟机无法直通内置摄像头，采用了手机采集图片的变通方案。因手机照片分辨率过高、打印棋盘格过小，OpenCV 未能成功检测棋盘格角点，标定未收敛。因此代码实现了参数容错机制，采用估算内参（fx=600, fy=600）完成了后续位姿解算任务。

### 2. AprilTag 检测与位姿解算结果
- 打印并测量了 tag36h11 ID 0 标签，黑色外框边长为 **0.1 米**。
- 演示截图：`apriltag_screenshot.png`（包含红X、绿Y、蓝Z三维坐标轴及距离数值）。
- 检测代码：`task2/test_single_image.py`，成功检测 Tag 并输出了旋转矩阵 R 和平移向量 t。

**运行结果**：
- 距离：0.1432 米
- 旋转矩阵 R：
  [[ 0.9998156   0.00453931  0.01865889]
   [-0.00501548  0.99966089  0.02555291]
   [-0.01853657 -0.02564179  0.99949932]]
