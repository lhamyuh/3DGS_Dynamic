Experiment V7 Log
Status: Failed (Baseline)

Key Specs: Spatial PE (L=6, 39D), Temporal PE (L=10, 21D), Radius=1.2.

Observations:

Final point count dropped from 1M to 33k (Over-pruning).

Jittering without squatting motion (Insufficient spatial room).

Black background and heavy noise (Low point density).

Action Plan: Evolve to V8 with Radius=1.5 and reduced initial opacity.


Experiment V8 Log
Status: Regressed

Training command (core):
-s ../dataset_dynamic/data/standup
-m output/Standup_V8_Smooth
--deformation_lr_init 0.001
--temporal_smoothness_weight 0.003

Observations:

Point count exploded to 444k at 30k iterations (very dense, unstable geometry support).

Mean deformation magnitude became too large (~9 to 10), causing blocky/fragmented appearance.

Visual result looked like a colored cube cloud instead of a clean human silhouette.

Root Cause Summary:

Deformation LR was too aggressive and densification remained permissive.

Temporal smoothness regularization alone could not constrain absolute motion drift.


Experiment V9 Log
Status: Improved but incomplete

Training profile: conservative deformation and densification controls (V9_Conservative).

Improvements:

SIBR point cloud now shows a clearer human subject with significantly better structure than V8.

Background noise reduced compared with V8.

Remaining Issues:

SIBR still shows a mostly squatting static posture and does not display live dynamic motion.

render_4d video is improved but still temporally discontinuous (not smooth enough).

A small amount of background noise remains.

Current understanding:

SIBR gaussian viewer loads point_cloud.ply states and does not consume deform_iter_*.pth at runtime.

Dynamic quality depends on both deformation magnitude control and temporal consistency beyond current settings.

Next Action Plan:

Keep conservative densification, further lower deformation drift, and increase temporal continuity constraints.

Add explicit dynamic-viewer path (or runtime deformation application) to bridge static SIBR display and 4D outputs.


Experiment V9.1 Log
Status: Good (current best)

Profile summary:

Based on V9 conservative training, integrated timestamp fix, temporal regularization knobs, and render-side anti-ghosting controls.

Key render settings (free camera):

--num_output_frames 420

--post_smooth_mode ema

--ema_alpha 0.9

--deform_time_samples 3

--deform_time_window 0.004

Observed improvements:

Fixed-camera sequence reaches target behavior: clear transition from squat toward standing with smooth temporal continuity.

Free-camera sequence also improved versus V9, with less tearing and better perceptual continuity.

Overall subjective quality is acceptable for current milestone.

Remaining limitations:

Minor residual ghosting still appears in some free-camera segments.

Occasional frame pacing drop is still visible in high-motion or high-detail transitions.

SIBR remains static per loaded point cloud state and does not natively replay deform_iter dynamic motion at runtime.

Next step candidates:

Tune free-camera interpolation and EMA aggressiveness per scene segment.

Evaluate a lightweight post-process denoise for residual background flicker.

If needed, design a viewer-side dynamic deformation integration path for true interactive 4D playback.


V9.1 Render Tuning (2026-04-29)

Fixed camera (clarity-first):

--lock_camera

--deform_time_samples 1

--deform_time_window 0.0

--deform_time_sigma 0.0

--post_smooth_mode none

Observation: details are slightly clearer, but frequent micro-jitter remains.

Free camera (slower motion, reduced ghosting):

--num_output_frames 720

--video_fps 24

--camera_time_samples 1

--camera_time_window 0.0

--camera_time_sigma 0.0

--deform_time_samples 3

--deform_time_window 0.003

--deform_time_sigma 0.0015

--post_smooth_mode ema

--ema_alpha 0.8

Observation: subject clarity improves and ghosting is reduced; motion feels slower, but residual ghosting still makes the motion appear faster than desired.


V9.1 Render Tuning Update (2026-04-29)

Free camera (current best, clarity-first):

--num_output_frames 840

--video_fps 24

--camera_time_samples 1

--camera_time_window 0.0

--camera_time_sigma 0.0

--deform_time_samples 1

--deform_time_window 0.0

--deform_time_sigma 0.0

--post_smooth_mode none

Observation: camera motion is smooth, subject is sharp, ghosting is nearly gone; slight background noise remains.

Fixed camera (current best, slight blur/noise):

--num_output_frames 420

--video_fps 30

--lock_camera

--deform_time_samples 2

--deform_time_window 0.001

--deform_time_sigma 0.005

--post_smooth_mode ema

--ema_alpha 0.7

Observation: jitter is reduced, but fine detail is slightly blurred and faint noise remains.


SIBR Viewer Modification Updated Log (2026-05-07)

Changes (P0 offline sequence playback):

- SIBR_viewers/src/projects/gaussianviewer/renderer/Config.hpp
	Added sequence playback CLI args: sequence_dir, sequence_fps, sequence_start, sequence_end, sequence_loop, sequence_pause.

- SIBR_viewers/src/projects/gaussianviewer/apps/gaussianViewer/main.cpp
	If sequence_dir is provided, pick the first .ply in the directory as the initial frame and pass sequence args to GaussianView.

- SIBR_viewers/src/projects/gaussianviewer/renderer/GaussianView.hpp/.cpp
	Added sequence state, file list, frame loading, fps timing, and GUI controls (pause/loop/fps/frame slider).
	Reuses GPU buffers when possible; reallocates if point count changes.

Purpose:

Enable dynamic point cloud playback from an offline PLY sequence without running the deformation network in the viewer.

Dynamic viewer launch (V9.1 PLY sequence):

\.\SIBR_gaussianViewer_app.exe -m E:\3dgsData\output\Standup_V9_1_SmoothDyn -s E:\3dgsData\output\Standup_V9_1_SmoothDyn --iteration 30000 --sequence_dir E:\3dgsData\output\Standup_V9_1_SmoothDyn\ply_sequence --sequence_fps 24 --sequence_loop

Optional params: --sequence_start 0 --sequence_end -1 --sequence_pause

Current viewing notes:

Static camera still shows slight blur and jitter; both static and free views show noticeable background noise.


V9.2 NoiseClean Best (2026-05-09)

Dataset: standup

Best config (Refine_Trial, 5k):

--iterations 5000
--deformation_lr_init 0.0002
--deformation_lr_gamma 0.9999
--densify_grad_threshold 0.00045
--densify_from_iter 1000
--densify_until_iter 10000
--prune_opacity_threshold 0.035
--opacity_reset_interval 1500
--temporal_smoothness_weight 0.004
--temporal_smoothness_start_iter 3000
--bg_consistency_weight 0.04
--bg_mask_weight 0.08
--bg_mask_threshold 0.05
--opacity_sparsity_weight 0.003
--opacity_sparsity_start_iter 500
--scale_reg_weight 0.0015
--scale_reg_start_iter 1000
--bbox_prune_scale 1.2
--bbox_prune_interval 400
--bbox_prune_start_iter 1500
--visibility_prune_min_count 1
--visibility_prune_interval 500
--visibility_prune_start_iter 1500

Metrics (train renders):

SSIM: 0.786222
PSNR: 19.612907
BG mean: 0.022170
BG max: 1.000000

Notes:

Best overall balance so far: lower background noise with higher SSIM/PSNR than stronger/strict trials.


V9.2 NoiseClean Conclusion (2026-05-10)

Status: Failed (noise cleanup regressed overall quality).

Summary:

- Subject details became blurred.
- Frame jitter increased.
- Halo around subject appeared.
- Noise became smeared/patchy rather than sparse.

Overall result is significantly worse than Standup_V9_1_SmoothDyn.

Next step:

Start a new optimization cycle from V9.3 baseline and re-evaluate noise strategy.


V9.3 NoisePrune (2026-05-10)

Status: Major improvement (noise greatly reduced, subject clarity preserved).

Training command:

/root/miniconda3/envs/4dgs_env/bin/python train.py
-s /autodl-fs/data/dataset_dynamic/data/standup
-m output/Standup_V9_3_NoisePrune
--iterations 30000
--deformation_lr_init 0.0002
--deformation_lr_gamma 0.9999
--densify_grad_threshold 0.00035
--densify_from_iter 1000
--densify_until_iter 12000
--prune_opacity_threshold 0.03
--opacity_reset_interval 1500
--temporal_smoothness_weight 0.003
--temporal_smoothness_start_iter 3000
--alpha_mask_mode full
--bg_consistency_weight 0.08
--opacity_sparsity_weight 0.0025
--opacity_sparsity_start_iter 500
--scale_reg_weight 0.0015
--scale_reg_start_iter 1000
--bbox_prune_scale 1.1
--bbox_prune_interval 400
--bbox_prune_start_iter 1500
--visibility_prune_min_count 3
--visibility_prune_interval 200
--visibility_prune_start_iter 800

Render (fixed camera):

/root/miniconda3/envs/4dgs_env/bin/python render_4d.py
-s /autodl-fs/data/dataset_dynamic/data/standup
-m output/Standup_V9_3_NoisePrune
--iteration 30000
--num_output_frames 420
--video_fps 30
--lock_camera
--deform_time_samples 2
--deform_time_window 0.001
--deform_time_sigma 0.0005
--post_smooth_mode ema
--ema_alpha 0.7

Render (free camera):

/root/miniconda3/envs/4dgs_env/bin/python render_4d.py
-s /autodl-fs/data/dataset_dynamic/data/standup
-m output/Standup_V9_3_NoisePrune
--iteration 30000
--num_output_frames 840
--video_fps 24
--deform_time_samples 2
--deform_time_window 0.001
--deform_time_sigma 0.0005
--post_smooth_mode ema
--ema_alpha 0.7

Progress note:

Compared with V9.1, background noise is largely removed and the subject remains sharp with stable motion.

Defects: Slight lens shake still exists, and details are slightly blurred.