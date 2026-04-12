import os
import numpy as np
from plyfile import PlyData, PlyElement


def convert_3dgs_filtered(ply_path, opacity_threshold=0.05):
    """
    带噪声过滤的转换脚本：
    1. 剔除不透明度低于 threshold 的点
    2. 将 SH 转换为标准 RGB
    """
    if not os.path.exists(ply_path):
        return

    print(f"正在读取并过滤模型: {ply_path}")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']

    # --- 过滤逻辑 ---
    # 提取不透明度并进行 Sigmoid 映射（3DGS 内部存储的是原始值）
    opacity = 1.0 / (1.0 + np.exp(-v['opacity']))

    # 只保留不透明度大于阈值的点，剔除那些“鲜艳的虚影”
    mask = opacity > opacity_threshold
    v_filtered = v[mask]
    print(f"原始点数: {len(v)}, 过滤后保留点数: {len(v_filtered)}")

    # --- 颜色转换 ---
    SH_C0 = 0.28209479177387814
    f_dc = np.stack([v_filtered['f_dc_0'], v_filtered['f_dc_1'], v_filtered['f_dc_2']], axis=-1)
    # 转换并限制在 0-1 范围，防止颜色“过爆”
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # --- 构造新 PLY ---
    new_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_data = np.empty(len(v_filtered), dtype=new_dtype)
    new_data['x'], new_data['y'], new_data['z'] = v_filtered['x'], v_filtered['y'], v_filtered['z']
    new_data['red'], new_data['green'], new_data['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    output_path = os.path.join(os.path.dirname(ply_path), "counter_clean_rgb.ply")
    PlyData([PlyElement.describe(new_data, 'vertex')]).write(output_path)
    print(f"清理完成！彩色模型已保存至: {output_path}")


# 执行过滤
target_file = "E:/3dgsData/output/counter_reproduce/point_cloud/iteration_30000/point_cloud.ply"
convert_3dgs_filtered(target_file, opacity_threshold=0.1)  # 增加阈值可进一步去噪