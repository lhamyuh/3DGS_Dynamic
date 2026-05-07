import os
import math
import numpy as np
import torch
from tqdm import tqdm

from scene import Scene
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from model.deform_model import DeformModel
from argparse import ArgumentParser


def _gaussian_weights(offsets, sigma):
    offsets = np.asarray(offsets, dtype=np.float32)
    sigma = float(max(sigma, 1e-6))
    weights = np.exp(-0.5 * np.square(offsets / sigma))
    weights_sum = float(np.sum(weights))
    if weights_sum <= 1e-8:
        return np.ones_like(offsets, dtype=np.float32) / float(len(offsets))
    return weights / weights_sum


def export_sequence(dataset, iteration, num_output_frames=300,
                    deform_time_samples=1, deform_time_window=0.0, deform_time_sigma=0.0,
                    sequence_dir=None, file_prefix="frame"):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        deform_model = DeformModel(D=8, W=256).cuda()
        deform_path = os.path.join(dataset.model_path, f"deform_iter_{iteration}.pth")
        if not os.path.exists(deform_path):
            print(f"Error: missing deformation weights {deform_path}")
            return
        deform_model.load_state_dict(torch.load(deform_path))
        deform_model.eval()

        views = scene.getTestCameras()
        if len(views) == 0:
            print("No cameras in TestSet, falling back to TrainSet...")
            views = scene.getTrainCameras()
        views = sorted(views, key=lambda x: x.timestamp)
        if len(views) == 0:
            print("Error: no cameras found. Check dataset path.")
            return

        t_min = float(views[0].timestamp)
        t_max = float(views[-1].timestamp)
        num_output_frames = int(max(2, num_output_frames))
        t_targets = np.linspace(t_min, t_max, num_output_frames)

        if sequence_dir is None:
            sequence_dir = os.path.join(dataset.model_path, "ply_sequence")
        os.makedirs(sequence_dir, exist_ok=True)

        base_xyz = gaussians.get_xyz.detach().clone()

        print(
            f"Exporting PLY sequence: frames={num_output_frames}, "
            f"t_range=[{t_min:.6f}, {t_max:.6f}], out={sequence_dir}"
        )

        for idx, t_cur in enumerate(tqdm(t_targets, desc="PLY Export")):
            if deform_time_samples > 1 and deform_time_window > 0.0:
                d_offsets = np.linspace(-deform_time_window, deform_time_window, int(deform_time_samples))
                d_sigma = deform_time_sigma if deform_time_sigma > 0.0 else max(deform_time_window * 0.5, 1e-6)
                d_weights = _gaussian_weights(d_offsets, d_sigma)
                d_xyz = torch.zeros_like(base_xyz)
                for off, weight in zip(d_offsets, d_weights):
                    t_sample = float(np.clip(t_cur + off, t_min, t_max))
                    d_xyz += deform_model(base_xyz, t_sample) * float(weight)
            else:
                d_xyz = deform_model(base_xyz, float(t_cur))

            gaussians._xyz.copy_(base_xyz + d_xyz)

            frame_name = f"{file_prefix}_{idx:05d}.ply"
            out_path = os.path.join(sequence_dir, frame_name)
            gaussians.save_ply(out_path)

            if idx % 50 == 0:
                torch.cuda.empty_cache()

        gaussians._xyz.copy_(base_xyz)
        print(f"PLY sequence saved to: {sequence_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Export 4DGS frames as a PLY sequence")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--num_output_frames", default=300, type=int)
    parser.add_argument("--deform_time_samples", default=1, type=int)
    parser.add_argument("--deform_time_window", default=0.0, type=float)
    parser.add_argument("--deform_time_sigma", default=0.0, type=float)
    parser.add_argument("--sequence_dir", default="", type=str)
    parser.add_argument("--file_prefix", default="frame", type=str)
    args = get_combined_args(parser)

    export_sequence(
        lp.extract(args),
        args.iteration,
        num_output_frames=args.num_output_frames,
        deform_time_samples=args.deform_time_samples,
        deform_time_window=args.deform_time_window,
        deform_time_sigma=args.deform_time_sigma,
        sequence_dir=args.sequence_dir if args.sequence_dir else None,
        file_prefix=args.file_prefix,
    )
