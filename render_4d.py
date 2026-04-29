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


def _smooth_frames_mean(frames, radius):
    if radius <= 0 or len(frames) <= 2:
        return frames
    smoothed = []
    for i in range(len(frames)):
        s = max(0, i - radius)
        e = min(len(frames), i + radius + 1)
        block = np.stack(frames[s:e], axis=0).astype(np.float32)
        smoothed.append(np.mean(block, axis=0).astype(np.uint8))
    return smoothed


def _smooth_frames_ema(frames, alpha):
    if len(frames) <= 1:
        return frames
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = [frames[0].astype(np.float32)]
    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32)
        out.append(alpha * out[-1] + (1.0 - alpha) * cur)
    return [np.clip(x, 0, 255).astype(np.uint8) for x in out]


def _gaussian_weights(offsets, sigma):
    offsets = np.asarray(offsets, dtype=np.float32)
    sigma = float(max(sigma, 1e-6))
    weights = np.exp(-0.5 * np.square(offsets / sigma))
    weights_sum = float(np.sum(weights))
    if weights_sum <= 1e-8:
        return np.ones_like(offsets, dtype=np.float32) / float(len(offsets))
    return weights / weights_sum


def _find_bracketing_views(views, t_value):
    view_a, view_b = None, None
    for i in range(len(views) - 1):
        if views[i].timestamp <= t_value <= views[i + 1].timestamp:
            view_a = views[i]
            view_b = views[i + 1]
            break
    if view_a is None:
        if len(views) == 0:
            return None, None
        view = views[0] if t_value <= views[0].timestamp else views[-1]
        return view, view
    return view_a, view_b


def _post_smooth_frames(frames, mode, radius, ema_alpha, lock_camera):
    mode = mode.lower()
    if mode == "none":
        return frames
    if mode == "mean":
        if not lock_camera:
            print("[Warning] mean post-smoothing in free-camera mode may cause ghosting; switching to ema.")
            return _smooth_frames_ema(frames, ema_alpha)
        return _smooth_frames_mean(frames, radius)
    if mode == "ema":
        return _smooth_frames_ema(frames, ema_alpha)
    if mode == "median":
        # median across temporal window to reduce double-exposure ghosting
        if radius <= 0:
            return frames
        med_smoothed = []
        for i in range(len(frames)):
            s = max(0, i - radius)
            e = min(len(frames), i + radius + 1)
            block = np.stack(frames[s:e], axis=0).astype(np.uint8)
            med_smoothed.append(np.median(block, axis=0).astype(np.uint8))
        return med_smoothed
    return frames


def render_video(dataset, iteration, pipeline, num_output_frames=300, lock_camera=False, smooth_radius=0,
                 post_smooth_mode="none", ema_alpha=0.85, video_fps=30,
                 deform_time_samples=1, deform_time_window=0.0,
                 deform_time_sigma=0.0,
                 camera_time_samples=1, camera_time_window=0.0,
                 camera_time_sigma=0.0,
                 ease_camera=True):
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

        t_min = float(views[0].timestamp)
        t_max = float(views[-1].timestamp)

        num_output_frames = int(max(2, num_output_frames))  # 目标总帧数
        t_targets = np.linspace(t_min, t_max, num_output_frames)

        print(
            f"正在进行平滑轨迹插值渲染，目标帧数: {num_output_frames}, "
            f"t范围: [{t_min:.6f}, {t_max:.6f}], fps: {video_fps}"
        )

        frames = []
        # 注意：这里直接循环时间戳，不再嵌套 views 循环
        for idx, t_cur in enumerate(tqdm(t_targets, desc="Smoothing Render")):
            # 1. 寻找相邻相机 (保持原逻辑)
            view_a_orig, view_b_orig = _find_bracketing_views(views, t_cur)
            if view_a_orig is None:
                print("错误：未在数据集中发现任何相机！请检查 -s 路径是否正确。")
                return

            # 2. 计算插值权重
            duration = view_b_orig.timestamp - view_a_orig.timestamp
            alpha = (t_cur - view_a_orig.timestamp) / duration if duration > 1e-7 else 0.0
            alpha = float(np.clip(alpha, 0.0, 1.0))
            if ease_camera:
                # Smoothstep interpolation reduces perceived stutter at keyframe boundaries.
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)

            # 3. 插值 R 和 T（或固定相机，仅让时间变化）
            if lock_camera:
                key_view = views[len(views) // 2]
                T_interp = key_view.T
                R_interp = key_view.R
                view_a_orig = key_view
                view_b_orig = key_view
            else:
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

            # 4. 计算 4D 形变（以当前时间为中心的可选时间平均）
            if deform_time_samples > 1 and deform_time_window > 0.0:
                d_offsets = np.linspace(-deform_time_window, deform_time_window, int(deform_time_samples))
                d_sigma = deform_time_sigma if deform_time_sigma > 0.0 else max(deform_time_window * 0.5, 1e-6)
                d_weights = _gaussian_weights(d_offsets, d_sigma)
                d_xyz = torch.zeros_like(gaussians.get_xyz)
                for off, weight in zip(d_offsets, d_weights):
                    t_sample = float(np.clip(t_cur + off, t_min, t_max))
                    d_xyz += deform_model(gaussians.get_xyz, t_sample) * float(weight)
            else:
                d_xyz = deform_model(gaussians.get_xyz, float(t_cur))
            means3D_deformed = gaussians.get_xyz + d_xyz

            # 5. 相机子帧采样与子帧合成（用于减少自由相机残影）
            subframes = []
            if camera_time_samples > 1 and camera_time_window > 0.0:
                c_offsets = np.linspace(-camera_time_window, camera_time_window, int(camera_time_samples))
                c_sigma = camera_time_sigma if camera_time_sigma > 0.0 else max(camera_time_window * 0.5, 1e-6)
                c_weights = _gaussian_weights(c_offsets, c_sigma)
            else:
                c_offsets = np.array([0.0])
                c_weights = np.array([1.0], dtype=np.float32)

            for coff, _weight in zip(c_offsets, c_weights):
                # 每个子帧仅改变相机位姿，复用已计算的 means3D_deformed（以降低计算量）
                t_cam = float(np.clip(t_cur + coff, t_min, t_max))

                # 重新计算插值权重与相机插值（与主循环相同逻辑，但以 t_cam 为中心）
                view_a_cam, view_b_cam = _find_bracketing_views(views, t_cam)
                if view_a_cam is None:
                    view_a_cam, view_b_cam = view_a_orig, view_b_orig
                duration_cam = view_b_cam.timestamp - view_a_cam.timestamp
                alpha_cam = (t_cam - view_a_cam.timestamp) / duration_cam if duration_cam > 1e-7 else 0.0
                alpha_cam = float(np.clip(alpha_cam, 0.0, 1.0))
                if ease_camera:
                    alpha_cam = alpha_cam * alpha_cam * (3.0 - 2.0 * alpha_cam)

                if lock_camera:
                    key_view = views[len(views) // 2]
                    T_c = key_view.T
                    R_c = key_view.R
                else:
                    T_c = view_a_cam.T * (1 - alpha_cam) + view_b_cam.T * alpha_cam
                    key_rots_c = R.from_matrix([view_a_cam.R.T, view_b_cam.R.T])
                    slerp_c = Slerp([0, 1], key_rots_c)
                    R_c = slerp_c([alpha_cam]).as_matrix()[0].T

                new_Rt_c = getWorld2View2(R_c, T_c)
                curr_view_c = curr_view
                curr_view_c.world_view_transform = torch.from_numpy(new_Rt_c).transpose(0, 1).cuda()
                curr_view_c.full_proj_transform = (curr_view_c.world_view_transform @ curr_view_c.projection_matrix)
                curr_view_c.camera_center = torch.inverse(curr_view_c.world_view_transform)[3, :3]

                render_pkg = render(curr_view_c, gaussians, pipeline, background, override_means3D=means3D_deformed)
                image = render_pkg["render"]
                img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                subframes.append(img_np)

            # 合成子帧为输出帧（高斯权重可减少边缘拖影）
            if len(subframes) == 1:
                out_frame = subframes[0]
            else:
                mode = post_smooth_mode.lower()
                if mode == "median":
                    out_frame = np.median(np.stack(subframes, axis=0), axis=0).astype(np.uint8)
                else:
                    subframe_stack = np.stack(subframes, axis=0).astype(np.float32)
                    weight_view = c_weights.astype(np.float32)[:, None, None, None]
                    out_frame = np.sum(subframe_stack * weight_view, axis=0).astype(np.uint8)

            frames.append(out_frame)
            cv2.imwrite(os.path.join(output_path, f"{idx:05d}.png"), out_frame)

            # 显存清理防止 5090 碎片堆积
            if idx % 50 == 0:
                torch.cuda.empty_cache()

        frames = _post_smooth_frames(frames, post_smooth_mode, smooth_radius, ema_alpha, lock_camera)

        # 5. 合成视频
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(
            os.path.join(dataset.model_path, "dynamic_result.mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'), int(max(1, video_fps)), (width, height)
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
    parser.add_argument("--num_output_frames", default=300, type=int)
    parser.add_argument("--lock_camera", action="store_true")
    parser.add_argument("--smooth_radius", default=0, type=int)
    parser.add_argument("--post_smooth_mode", default="none", choices=["none", "mean", "ema", "median"], type=str)
    parser.add_argument("--ema_alpha", default=0.85, type=float)
    parser.add_argument("--video_fps", default=30, type=int)
    parser.add_argument("--deform_time_samples", default=1, type=int)
    parser.add_argument("--deform_time_window", default=0.0, type=float)
    parser.add_argument("--deform_time_sigma", default=0.0, type=float)
    parser.add_argument("--camera_time_samples", default=1, type=int)
    parser.add_argument("--camera_time_window", default=0.0, type=float)
    parser.add_argument("--camera_time_sigma", default=0.0, type=float)
    parser.add_argument("--disable_ease_camera", action="store_true")
    args = get_combined_args(parser)

    render_video(
        lp.extract(args),
        args.iteration,
        pp.extract(args),
        num_output_frames=args.num_output_frames,
        lock_camera=args.lock_camera,
        smooth_radius=args.smooth_radius,
        post_smooth_mode=args.post_smooth_mode,
        ema_alpha=args.ema_alpha,
        video_fps=args.video_fps,
        deform_time_samples=args.deform_time_samples,
        deform_time_window=args.deform_time_window,
        deform_time_sigma=args.deform_time_sigma,
        camera_time_samples=args.camera_time_samples,
        camera_time_window=args.camera_time_window,
        camera_time_sigma=args.camera_time_sigma,
        ease_camera=not args.disable_ease_camera,
    )