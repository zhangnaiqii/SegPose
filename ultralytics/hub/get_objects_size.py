import os
import glob
import numpy as np
from tqdm import tqdm


def analyze_yolo_sizes(label_dir, img_size=640):
    """
    分析YOLO标注文件中的目标尺寸分布 (Small/Medium/Large)

    Args:
        label_dir (str): 标注文件夹路径 (.txt文件)
        img_size (int): 模型输入的图像尺寸，默认为 640 (COCO标准)
    """

    # COCO 标准阈值
    area_threshold_small = 32 ** 2  # 1024
    area_threshold_large = 96 ** 2  # 9216

    cnt_small = 0
    cnt_medium = 0
    cnt_large = 0
    total_objects = 0

    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

    if not txt_files:
        print(f"❌ 错误: 在 {label_dir} 没有找到 .txt 文件")
        return

    print(f"🔍 正在分析 {len(txt_files)} 个标注文件...")

    for txt_file in tqdm(txt_files):
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO 格式: class x_center y_center width height ...
            # 都是归一化到 0-1 的
            try:
                # 只需要 w 和 h (索引 3 和 4)
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # 还原为像素尺寸 (基于 img_size)
                w_pix = w_norm * img_size
                h_pix = h_norm * img_size

                area = w_pix * h_pix

                if area < area_threshold_small:
                    cnt_small += 1
                elif area > area_threshold_large:
                    cnt_large += 1
                else:
                    cnt_medium += 1

                total_objects += 1

            except ValueError:
                continue

    if total_objects == 0:
        print("没有检测到任何目标。")
        return

    # 计算比例
    ratio_s = cnt_small / total_objects * 100
    ratio_m = cnt_medium / total_objects * 100
    ratio_l = cnt_large / total_objects * 100

    print("\n" + "=" * 40)
    print(f"📊 数据集尺寸分布分析 (基于输入尺寸 {img_size}x{img_size})")
    print("=" * 40)
    print(f"📦 总目标数量: {total_objects}")
    print("-" * 40)
    print(f"🔴 Small  (P3, < 32²): \t{cnt_small}\t ({ratio_s:.2f}%)")
    print(f"🟢 Medium (P4, 32²-96²): \t{cnt_medium}\t ({ratio_m:.2f}%)")
    print(f"🔵 Large  (P5, > 96²): \t{cnt_large}\t ({ratio_l:.2f}%)")
    print("=" * 40)

    # 简单的建议
    print("\n💡 架构调整建议:")
    if ratio_l < 5:
        print("👉 P5 (Large) 占比极低。可以考虑移除 Layer 18-20 (P5 Head)，仅使用 P3+P4。")
        print("   这将解决你之前遇到的深层梯度冲突问题，并大幅加速训练。")
    elif ratio_s < 5:
        print("👉 P3 (Small) 占比极低。可以考虑移除 P3 Head，仅使用 P4+P5。")
    else:
        print("👉 数据分布较均衡，建议保留 P3/P4/P5 全部分支。")


# ================= 使用示例 =================
# 请将下面的路径替换为您数据集的 labels/train 文件夹路径
# 比如: r"D:\Datasets\Blueberry\labels\train"
target_folder = "Z:/datasets/segment/straw_segpose/pose/labels/train"

# 如果您还没准备好路径，可以先不运行，把代码复制走
analyze_yolo_sizes(target_folder, img_size=640)
