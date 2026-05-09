V9.1_to_V9.3_Noise_Cleanup_Log (2026-05-10)

Goal
- Remove floating background noise while keeping the subject sharp and stable.

Baseline (V9.1)
- Subject detail: good.
- Noise: sparse but persistent floating points around the subject.
- Render tuning helped with motion but did not remove noise at the source.

What went wrong in V9.2 (NoiseClean)
- Changes focused on stronger regularizers and prune thresholds.
- Result: subject became blurred, noise turned into smeared patches, and motion jitter increased.
- Conclusion: aggressive smoothing and partial background supervision caused quality regression.

Root cause analysis
- Background points were not fully supervised in the main L1/SSIM loss.
- Random points that rarely appear were not pruned early enough and survived into later training.
- Render-side smoothing can hide noise but also smears subject details, making results worse.

Key changes that enabled V9.3 improvements
1) Full background supervision with alpha masks
   - Added alpha_mask_mode=full so background is included in L1/SSIM.
   - This makes background noise immediately costly and pushes it out during training.

2) Stronger background consistency
   - bg_consistency_weight increased to 0.08.
   - Enforces uniform background color where alpha indicates background.

3) Early, strict visibility pruning
   - visibility_prune_min_count=3, interval=200, start=800.
   - Removes points that only appear occasionally (typical floating noise).

4) Tighter spatial pruning
   - bbox_prune_scale=1.1 to keep points closer to the subject.

5) Moderate regularizers to avoid blur
   - temporal_smoothness_weight=0.003.
   - opacity_sparsity_weight=0.0025.
   - Kept enough detail while still suppressing noise.

V9.3 training command (30k)
/root/miniconda3/envs/4dgs_env/bin/python train.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_NoisePrune \
--iterations 30000 \
--deformation_lr_init 0.0002 \
--deformation_lr_gamma 0.9999 \
--densify_grad_threshold 0.00035 \
--densify_from_iter 1000 \
--densify_until_iter 12000 \
--prune_opacity_threshold 0.03 \
--opacity_reset_interval 1500 \
--temporal_smoothness_weight 0.003 \
--temporal_smoothness_start_iter 3000 \
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

V9.3 render commands
Fixed camera:
/root/miniconda3/envs/4dgs_env/bin/python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_NoisePrune --iteration 30000 \
--num_output_frames 420 --video_fps 30 --lock_camera \
--deform_time_samples 2 --deform_time_window 0.001 --deform_time_sigma 0.0005 \
--post_smooth_mode ema --ema_alpha 0.7

Free camera:
/root/miniconda3/envs/4dgs_env/bin/python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_NoisePrune --iteration 30000 \
--num_output_frames 840 --video_fps 24 \
--deform_time_samples 2 --deform_time_window 0.001 --deform_time_sigma 0.0005 \
--post_smooth_mode ema --ema_alpha 0.7

Optional render cleanup (alpha-based background removal)
- Use render_4d with alpha-based background cleanup to remove remaining noise
  without blurring the subject.

Trial metrics (5k)
- SSIM: 0.896984
- PSNR: 19.752190
- BG mean: 0.011692

Outcome summary
- Background noise largely removed at training time.
- Subject remains sharp with stable motion.
- This is a major improvement over V9.1 and V9.2.

Open issues
- Slight lens shake remains in some sequences.
- Very mild blur in high-motion frames (can be tuned by render settings).
