import os
import numpy as np
from plyfile import PlyData, PlyElement


def convert_3dgs_pure_color(ply_path):
    """
    纯净转换脚本：
    1. 100% 保留所有点，不进行任何透明度或空间过滤
    2. 仅将 f_dc (SH degree 0) 转换为标准 RGB
    """
    if not os.path.exists(ply_path):
        print(f"错误：找不到文件 {ply_path}")
        return

    print(f"正在读取原始点云: {ply_path}")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    num_pts = len(v)

    # --- 颜色转换逻辑 (不进行 Mask 过滤) ---
    SH_C0 = 0.28209479177387814
    # 提取球谐系数的直流分量 (f_dc)
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)

    # 转换公式：RGB = 0.5 + SH_C0 * f_dc
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # --- 构造新 PLY (保持原始点数) ---
    # 定义新数据结构，只包含坐标和颜色
    new_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]
    new_data = np.empty(num_pts, dtype=new_dtype)

    # 直接赋值，不经过任何索引过滤
    new_data['x'] = v['x']
    new_data['y'] = v['y']
    new_data['z'] = v['z']
    new_data['red'] = rgb[:, 0]
    new_data['green'] = rgb[:, 1]
    new_data['blue'] = rgb[:, 2]

    # 保存路径
    output_path = os.path.join(os.path.dirname(ply_path), "full_points_rgb.ply")

    print(f"正在保存完整点云，总点数: {num_pts}")
    PlyData([PlyElement.describe(new_data, 'vertex')]).write(output_path)
    print(f"转换完成！完整彩色模型已保存至: {output_path}")


# 执行转换
target_file = "E:/3dgsData/output/Standup_V6_FullAction/30000/point_cloud.ply"
convert_3dgs_pure_color(target_file)