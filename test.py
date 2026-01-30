import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFrame, QCheckBox,
                             QProgressBar, QGraphicsOpacityEffect)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPoint
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont

# --- 样式表 (QSS) ---
# 这里定义了软件的“皮肤”：深色背景、圆角、霓虹绿/红配色
STYLESHEET = """
QMainWindow {
    background-color: #121212;
}
QFrame#Panel {
    background-color: #1E1E1E;
    border-radius: 15px;
    border: 1px solid #333;
}
QLabel {
    color: #E0E0E0;
    font-family: 'Segoe UI', sans-serif;
}
QLabel#Title {
    font-size: 16px;
    font-weight: bold;
    color: #AAAAAA;
    padding-bottom: 10px;
}
/* 模拟霓虹开关按钮 */
QCheckBox {
    color: white;
    font-size: 14px;
    padding: 5px;
}
QCheckBox::indicator {
    width: 40px;
    height: 20px;
    border-radius: 10px;
    background-color: #333;
}
QCheckBox::indicator:checked {
    background-color: #00E676; /* 霓虹绿 */
    border: 1px solid #00FF88;
}
/* 统计数字 */
QLabel#StatNumber {
    font-size: 36px;
    font-weight: bold;
    color: #00E676;
}
/* 详细信息文本 */
QLabel#DetailText {
    font-family: 'Consolas', monospace;
    color: #00B0FF;
    font-size: 12px;
}
"""


class VideoLabel(QLabel):
    """自定义的Label，用于捕获鼠标在视频上的移动位置"""
    mouse_moved = pyqtSignal(QPoint)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)  # 开启鼠标追踪，不需要按下也能捕获移动
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #000; border-radius: 10px;")

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.pos())


class AgriVisionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AgriVision Pro v2.0")
        self.resize(1280, 720)
        self.setStyleSheet(STYLESHEET)

        # 核心数据状态
        self.is_inspecting = False  # 是否开启质检
        self.is_counting = False  # 是否开启计数
        self.is_pose = False  # 是否开启姿态
        self.current_mouse_pos = None

        # 主布局容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === 左侧栏：功能控制 ===
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("MODULES", objectName="Title"))

        # 功能开关
        self.cb_detect = QCheckBox("1. 目标定位 (Localization)")
        self.cb_detect.setChecked(True)  # 默认开启

        self.cb_inspect = QCheckBox("2. 智能质检 (Quality)")
        self.cb_inspect.toggled.connect(lambda: setattr(self, 'is_inspecting', self.cb_inspect.isChecked()))

        self.cb_count = QCheckBox("3. 实时计数 (Counting)")
        self.cb_count.toggled.connect(lambda: setattr(self, 'is_counting', self.cb_count.isChecked()))

        self.cb_pose = QCheckBox("4. 姿态与点云 (Pose)")
        self.cb_pose.toggled.connect(lambda: setattr(self, 'is_pose', self.cb_pose.isChecked()))

        left_layout.addWidget(self.cb_detect)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.cb_inspect)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.cb_count)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.cb_pose)
        left_layout.addStretch()  # 顶上去

        # === 中间区域：视频可视化 ===
        center_panel = QFrame()
        center_panel.setObjectName("Panel")
        center_layout = QVBoxLayout(center_panel)
        center_layout.addWidget(QLabel("MAIN VISUALIZATION", objectName="Title"))

        self.video_display = VideoLabel()
        self.video_display.mouse_moved.connect(self.update_hover_info)
        center_layout.addWidget(self.video_display)

        # === 右侧栏：统计与交互 ===
        right_panel = QFrame()
        right_panel.setObjectName("Panel")
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)

        # 上部：统计
        right_layout.addWidget(QLabel("STATISTICS", objectName="Title"))
        self.lbl_count = QLabel("1,245")
        self.lbl_count.setObjectName("StatNumber")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(QLabel("Total Count:", alignment=Qt.AlignmentFlag.AlignCenter))
        right_layout.addWidget(self.lbl_count)

        # 模拟一个环形图（用进度条代替演示）
        right_layout.addWidget(QLabel("Pass Rate: 92.5%", alignment=Qt.AlignmentFlag.AlignCenter))
        self.progress = QProgressBar()
        self.progress.setValue(92)
        self.progress.setStyleSheet("QProgressBar::chunk { background-color: #00E676; }")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        right_layout.addWidget(self.progress)

        right_layout.addSpacing(30)

        # 下部：详细信息交互区
        right_layout.addWidget(QLabel("SELECTED TARGET INFO", objectName="Title"))
        self.info_panel = QLabel("Waiting for selection...")
        self.info_panel.setObjectName("DetailText")
        self.info_panel.setWordWrap(True)
        self.info_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        right_layout.addWidget(self.info_panel, stretch=1)

        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, stretch=1)  # 中间拉伸
        main_layout.addWidget(right_panel)

        # === 模拟视频流定时器 ===
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.timer.start(30)  # 30ms 刷新一次，约30fps

        # 模拟数据
        self.fake_objects = []  # 存储模拟的果实坐标

    def update_video_frame(self):
        """模拟OpenCV处理流程：获取帧 -> 绘图 -> 显示"""
        # 1. 创建一个空黑帧 (模拟摄像头画面)
        h, w = 600, 800
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # 深灰色背景，模拟传送带

        # 2. 模拟生成一些移动的果实
        self.simulate_fruits(w, h)

        # 3. 绘制逻辑 (根据左侧开关决定画什么)
        for obj in self.fake_objects:
            x, y, radius, is_bad = obj['x'], obj['y'], obj['r'], obj['bad']

            # --- 模块1：目标定位 (基础框/圆) ---
            if self.cb_detect.isChecked():
                color = (0, 255, 0)  # 默认绿色
                if self.is_inspecting and is_bad:
                    color = (0, 0, 255)  # 质检开启且坏果，变红

                cv2.rectangle(frame, (x - radius, y - radius), (x + radius, y + radius), color, 2)
                label = "DEFECT" if (self.is_inspecting and is_bad) else "PASS"
                cv2.putText(frame, label, (x - radius, y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # --- 模块2：掩码 (简化为填充) ---
            # 实际项目中这里是实例分割掩码覆盖
            # 这里简单演示：质检开启时，稍微填充一下颜色
            if self.is_inspecting:
                overlay = frame.copy()
                fill_color = (0, 0, 255) if is_bad else (0, 255, 0)
                cv2.circle(overlay, (x, y), radius, fill_color, -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # --- 模块4：姿态与点云 ---
            if self.is_pose:
                # 画轴向 (X, Y, Z)
                axis_len = 40
                # X轴(红), Y轴(绿), Z轴(蓝)
                cv2.arrowedLine(frame, (x, y), (x + axis_len, y), (0, 0, 255), 2)
                cv2.arrowedLine(frame, (x, y), (x, y - axis_len), (0, 255, 0), 2)
                cv2.arrowedLine(frame, (x, y), (x + 15, y - 15), (255, 0, 0), 2)

                # 画模拟点云 (随机噪点)
                for _ in range(10):
                    px = x + np.random.randint(-radius, radius)
                    py = y + np.random.randint(-radius, radius)
                    if (px - x) ** 2 + (py - y) ** 2 < radius ** 2:
                        frame[py, px] = (255, 255, 255)

        # 4. 转换格式并显示在Qt Label上
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # 保持比例缩放适应窗口
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_display.setPixmap(scaled_pixmap)

        # 保存当前的缩放比例，用于鼠标坐标映射
        self.img_scale_w = scaled_pixmap.width()
        self.img_scale_h = scaled_pixmap.height()

    def simulate_fruits(self, w, h):
        """简单的物理模拟，让果实动起来"""
        if not hasattr(self, 'frame_count'): self.frame_count = 0
        self.frame_count += 1

        # 每50帧生成一个新果实
        if self.frame_count % 50 == 0:
            self.fake_objects.append({
                'x': 50, 'y': h // 2, 'r': 40,
                'bad': np.random.rand() > 0.8,  # 20%概率坏果
                'id': np.random.randint(1000, 9999)
            })

        # 移动果实
        for obj in self.fake_objects:
            obj['x'] += 5  # 向右移动

        # 移除移出屏幕的
        self.fake_objects = [obj for obj in self.fake_objects if obj['x'] < w + 100]

    def update_hover_info(self, pos):
        """处理鼠标悬停逻辑"""
        # 需要将 UI 上的鼠标坐标 映射回 视频帧的坐标
        # 这里为了简化代码，直接做一个简单的距离判定演示

        # 获取显示区域的偏移（因为KeepAspectRatio会产生留白）
        label_w = self.video_display.width()
        label_h = self.video_display.height()
        pix_w = getattr(self, 'img_scale_w', label_w)
        pix_h = getattr(self, 'img_scale_h', label_h)

        offset_x = (label_w - pix_w) // 2
        offset_y = (label_h - pix_h) // 2

        # 映射坐标 (近似计算，实际需要更严谨的矩阵变换)
        scale_x = 800 / pix_w  # 原图宽800
        scale_y = 600 / pix_h  # 原图高600

        real_x = (pos.x() - offset_x) * scale_x
        real_y = (pos.y() - offset_y) * scale_y

        # 查找鼠标是否碰到了果实
        found = False
        for obj in self.fake_objects:
            dist = np.sqrt((real_x - obj['x']) ** 2 + (real_y - obj['y']) ** 2)
            if dist < obj['r']:
                # 找到了！更新右侧信息
                status = "DEFECT (坏果)" if obj['bad'] else "PASS (合格)"
                color_code = "#FF5252" if obj['bad'] else "#00E676"

                info_html = f"""
                ID: <span style='color:white'>#{obj['id']}</span><br><br>
                Status: <span style='color:{color_code}; font-weight:bold'>{status}</span><br><br>
                Position: [{int(obj['x'])}, {int(obj['y'])}]<br>
                Axial Pitch: 45°<br>
                Axial Yaw: 12°<br>
                """
                self.info_panel.setText(info_html)
                self.info_panel.setStyleSheet(f"border: 1px solid {color_code}; border-radius: 5px; padding: 10px;")
                found = True
                break

        if not found:
            self.info_panel.setText("Waiting for selection...<br>(Hover over a fruit)")
            self.info_panel.setStyleSheet("border: none; padding: 10px;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgriVisionUI()
    window.show()
    sys.exit(app.exec())