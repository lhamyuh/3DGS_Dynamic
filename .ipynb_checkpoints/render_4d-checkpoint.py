import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from scene import Scene
from os import makedirs
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from model.deform_model import DeformModel
from argparse import ArgumentParser


def render_video(dataset, iteration, pipeline):
    with torch.no_grad():
        # 1. 加载基础高斯模型
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 2. 加载变形场模型 (DeformModel)
        # 注意：D=4, W=256 是 4DGS 的默认参数，请根据你模型定义确认
        deform_model = DeformModel(D=4, W=256).cuda()
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



        frames = []
        for idx, view in enumerate(tqdm(views, desc="Rendering Progress")):
            # 【核心逻辑】计算该时刻 t 的位移
            # view.fid 或 view.timestamp 记录了当前帧的时间
            timestamp_value = float(view.timestamp)
            d_xyz = deform_model(gaussians.get_xyz, timestamp_value)

            # 得到变形后的坐标
            means3D_deformed = gaussians.get_xyz + d_xyz

            # 调用你修改过的支持 override_means3D 的 render 函数
            render_pkg = render(view, gaussians, pipeline, background, override_means3D=means3D_deformed)
            image = render_pkg["render"]

            # 转换为 numpy 格式用于合成视频
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            frames.append(img_np)

            # 同时保存单帧图片
            cv2.imwrite(os.path.join(output_path, f"{idx:05d}.png"), img_np)

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