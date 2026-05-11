LEGO Experiment Log
Date: 2026-05-12

Goal
- Reduce visible jitter while keeping subject detail sharp.
- Keep background noise low (alpha-supervised pruning).

Dataset
- /autodl-fs/data/dataset_dynamic/data/lego

Baseline (V9.3.1, 30k)
Training command:
/root/miniconda3/envs/4dgs_env/bin/python train.py \
-s /autodl-fs/data/dataset_dynamic/data/lego \
-m output/Lego_V9_3_1_NoisePrune \
--iterations 30000 \
--deformation_lr_init 0.0002 \
--deformation_lr_gamma 0.9999 \
--densify_grad_threshold 0.00035 \
--densify_from_iter 1000 \
--densify_until_iter 12000 \
--prune_opacity_threshold 0.03 \
--opacity_reset_interval 1500 \
--temporal_smoothness_weight 0.004 \
--temporal_smoothness_start_iter 2000 \
--temporal_smoothness_epsilon 0.004 \
--alpha_mask_mode full \
--bg_consistency_weight 0.08 \
--opacity_sparsity_weight 0.0025 \
--opacity_sparsity_start_iter 500 \
--scale_reg_weight 0.0015 \
--scale_reg_start_iter 1000 \
--bbox_prune_scale 1.1 \
--bbox_prune_interval 400 \
--bbox_prune_start_iter 1500 \
--visibility_prune_min_count 3 \
--visibility_prune_interval 200 \
--visibility_prune_start_iter 800

Render baseline (fixed camera):
/root/miniconda3/envs/4dgs_env/bin/python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/lego \
-m output/Lego_V9_3_1_NoisePrune --iteration 30000 \
--num_output_frames 420 --video_fps 30 --lock_camera \
--deform_time_samples 2 --deform_time_window 0.001 --deform_time_sigma 0.0005 \
--post_smooth_mode ema --ema_alpha 0.7

Render-only attempts (no training change)
- Strong EMA smoothing:
  --deform_time_samples 5 --deform_time_window 0.003 --deform_time_sigma 0.0015 --post_smooth_mode ema --ema_alpha 0.85
  Result: jitter still visible; blur increases.
- 4x interpolation + median:
  --num_output_frames 1680 --video_fps 120 --post_smooth_mode median --smooth_radius 2
  Result: higher frame rate but jitter still visible.

Conclusion: jitter is not solved by render smoothing alone.

Trials (5k) and results

V9.3.2 NoisePrune Smooth Trial
Command:
... output/Lego_V9_3_2_NoisePrune_Smooth_Trial
Key changes:
- temporal_smoothness_weight 0.006
- temporal_smoothness_start_iter 1000
- temporal_smoothness_epsilon 0.002
Metrics (train renders):
- SSIM 0.786081
- PSNR 18.283075
- BG mean 0.009366
- BG max 1.000000
Notes:
- Noise slightly lower but subject detail degraded; jitter still visible.

V9.3.3 NoisePrune Smooth Trial
Command:
... output/Lego_V9_3_3_NoisePrune_Smooth_Trial
Key changes:
- temporal_smoothness_weight 0.0045
- temporal_smoothness_start_iter 2000
- temporal_smoothness_epsilon 0.006
Metrics (train renders):
- SSIM 0.786804
- PSNR 18.211512
- BG mean 0.009470
- BG max 1.000000
Notes:
- No meaningful improvement over V9.3.2.

V9.3.4 NoisePrune Smooth Trial
Command:
... output/Lego_V9_3_4_NoisePrune_Smooth_Trial
Key changes:
- temporal_smoothness_weight 0.005
- temporal_smoothness_start_iter 2500
- temporal_smoothness_epsilon 0.008
Metrics (train renders):
- SSIM 0.786446
- PSNR 18.244844
- BG mean 0.009468
- BG max 1.000000
Notes:
- No improvement in jitter; clarity not better than 9.3.1.

V9.3.5 TimeCons Trial (multi-sample temporal smoothing)
Command:
... output/Lego_V9_3_5_TimeCons_Trial
Key changes:
- temporal_smoothness_samples 5
- temporal_smoothness_weight 0.004
- temporal_smoothness_start_iter 2000
- temporal_smoothness_epsilon 0.006
Metrics (train renders):
- SSIM 0.784468
- PSNR 18.185883
- BG mean 0.009645
- BG max 1.000000
Notes:
- Clarity worse; jitter not solved.

Summary (so far)
- V9.3.1 remains the clearest baseline.
- Increasing temporal smoothing weight or multi-sample smoothing reduces clarity but does not eliminate jitter.
- Render-only smoothing is insufficient to fix motion instability.

Next ideas
- Revisit deformation stability (learning rate schedule or regularization in deformation network).
- Consider adding a velocity or acceleration penalty instead of direct position smoothing.

Detail-preserving jitter reduction ideas (training-side)
- Velocity loss (preferred): penalize d_xyz(t+eps) - d_xyz(t) / eps with a small weight. This targets high-frequency jitter without blurring static detail.
- Acceleration loss: penalize d_xyz(t+eps) - 2*d_xyz(t) + d_xyz(t-eps). Stronger jitter suppression, but weight must be tiny to avoid motion dulling.
- Opacity/scale temporal consistency: small L1 on opacity and scaling between t and t+eps to reduce flicker without smearing color detail.
- Deformation LR schedule: reduce deformation_lr_init or accelerate decay (lower gamma) after mid-iterations to stabilize motion once geometry is formed.
- Densify cutoff earlier: stop densification sooner (or raise densify_grad_threshold) to reduce late unstable points that amplify jitter.
- Motion outlier pruning: prune points with unusually large temporal displacement (e.g., |d_xyz(t+eps) - d_xyz(t)| above a threshold).

Detail-preserving jitter reduction ideas (render-side)
- Use deform_time_samples > 1 with small window, but keep post_smooth_mode none or median to avoid blur.
- Avoid strong EMA; prefer median smoothing with small radius for motion stability without smearing edges.
