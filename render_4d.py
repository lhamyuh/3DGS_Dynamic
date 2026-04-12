import torch
import os
import cv2
import math
import numpy as np
from tqdm import tqdm
from scene import Scene
from os import makedirs
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from model.deform_model import DeformModel
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def render_video(dataset, iteration, pipeline):
    with torch.no_grad():
        # 1. 加载基础高斯模型
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 2. 加载变形场模型 (DeformModel)
        # 注意：D=8, W=256 是 4DGS 的默认参数，请根据你模型定义确认
        deform_model = DeformModel(D=8, W=256).cuda()
        deform_path = os.path.join(dataset.model_path, f"deform_iter_{iteration}.pth")

        if not os.path.exists(deform_path):
            print(f"错误：找不到变形权重文件 {deform_path}")
            return

        deform_model.load_state_dict(torch.load(deform_path))
        deform_model.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 3. 创建输出目录
        output_path = os.path.join(dataset.model_path, "render_4d_video")
        makedirs(output_path, exist_ok=True)

        # 4. 获取测试视角（通常包含完整的时间序列）
        views = scene.getTestCameras()
        if len(views) == 0:
            print("未在 TestSet 中发现相机，正在尝试从 TrainSet 获取...")
            views = scene.getTrainCameras()

        # 按时间戳排序，确保视频动作连贯
        views = sorted(views, key=lambda x: x.timestamp)

        if len(views) == 0:
            print("错误：未在数据集中发现任何相机！请检查 -s 路径是否正确。")
            return
        print(f"正在渲染 {len(views)} 帧动态画面...")

        num_output_frames = 300  # 目标总帧数
        t_targets = np.linspace(0.0, 1.0, num_output_frames)

        print(f"正在进行平滑轨迹插值渲染，目标帧数: {num_output_frames}")

        frames = []
        # 注意：这里直接循环时间戳，不再嵌套 views 循环
        for idx, t_cur in enumerate(tqdm(t_targets, desc="Smoothing Render")):
            # 1. 寻找相邻相机 (保持原逻辑)
            view_a_orig, view_b_orig = None, None
            for i in range(len(views) - 1):
                if views[i].timestamp <= t_cur <= views[i + 1].timestamp:
                    view_a_orig = views[i]
                    view_b_orig = views[i + 1]
                    break
            if view_a_orig is None:
                view_a_orig = views[0] if t_cur <= views[0].timestamp else views[-1]
                view_b_orig = view_a_orig

            # 2. 计算插值权重
            duration = view_b_orig.timestamp - view_a_orig.timestamp
            alpha = (t_cur - view_a_orig.timestamp) / duration if duration > 1e-7 else 0.0

            # 3. 插值 R 和 T
            T_interp = view_a_orig.T * (1 - alpha) + view_b_orig.T * alpha
            key_rots = R.from_matrix([view_a_orig.R.T, view_b_orig.R.T])
            slerp = Slerp([0, 1], key_rots)
            R_interp = slerp([alpha]).as_matrix()[0].T

            # --- 【关键修复：手动重构临时相机对象】 ---
            from argparse import Namespace
            from utils.graphics_utils import getWorld2View2

            # 创建一个完全独立于 views 列表的临时命名空间
            curr_view = Namespace()

            # 1. 基础维度与时间
            curr_view.image_height = int(view_a_orig.image_height)
            curr_view.image_width = int(view_a_orig.image_width)
            curr_view.timestamp = t_cur

            # 2. 内参属性 (根据报错，render 函数在读取 FoVx 和 FoVy)
            curr_view.FoVx = view_a_orig.FoVx
            curr_view.FoVy = view_a_orig.FoVy

            # 如果 render 函数还读取了 tanfov，我们也一并补齐（预防下个报错）
            if hasattr(view_a_orig, 'tanfovX'):
                curr_view.tanfovX = view_a_orig.tanfovX
                curr_view.tanfovY = view_a_orig.tanfovY
            else:
                import math
                curr_view.tanfovX = math.tan(view_a_orig.FoVx * 0.5)
                curr_view.tanfovY = math.tan(view_a_orig.FoVy * 0.5)

            # 核心矩阵重算
            new_Rt = getWorld2View2(R_interp, T_interp)
            curr_view.world_view_transform = torch.from_numpy(new_Rt).transpose(0, 1).cuda()
            curr_view.projection_matrix = view_a_orig.projection_matrix.cuda()
            curr_view.full_proj_transform = (curr_view.world_view_transform @ curr_view.projection_matrix)
            curr_view.camera_center = torch.inverse(curr_view.world_view_transform)[3, :3]

            # 4. 计算 4D 形变
            t_tensor = torch.full((1,), t_cur).cuda()
            d_xyz = deform_model(gaussians.get_xyz, t_tensor)
            means3D_deformed = gaussians.get_xyz + d_xyz

            # 5. 执行渲染 (使用 curr_view 而不是 view_a_orig)
            render_pkg = render(curr_view, gaussians, pipeline, background, override_means3D=means3D_deformed)
            image = render_pkg["render"]

            # 6. 后处理与保存
            img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            frames.append(img_np)
            cv2.imwrite(os.path.join(output_path, f"{idx:05d}.png"), img_np)

            # 显存清理防止 5090 碎片堆积
            if idx % 50 == 0:
                torch.cuda.empty_cache()

        # 5. 合成视频
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(
            os.path.join(dataset.model_path, "dynamic_result.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height)
        )
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        print(f"视频已保存至: {os.path.join(dataset.model_path, 'dynamic_result.mp4')}")


if __name__ == "__main__":
    parser = ArgumentParser(description="4DGS 推理脚本")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    args = get_combined_args(parser)

    render_video(lp.extract(args), args.iteration, pp.extract(args))