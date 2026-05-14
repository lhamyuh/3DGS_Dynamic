全部开启基线
python train.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/StandUp_V9_3_NoisePrune_White \
--iterations 30000 \
--white_background \
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

无时序正则
python train.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/StandUp_V9_3_NoisePrune_White_NoTS \
--iterations 30000 \
--white_background \
--deformation_lr_init 0.0002 \
--deformation_lr_gamma 0.9999 \
--densify_grad_threshold 0.00035 \
--densify_from_iter 1000 \
--densify_until_iter 12000 \
--prune_opacity_threshold 0.03 \
--opacity_reset_interval 1500 \
--temporal_smoothness_weight 0.0 \
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

无背景一致性
python train.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/StandUp_V9_3_NoisePrune_White_NoBG \
--iterations 30000 \
--white_background \
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
--bg_consistency_weight 0.0 \
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

无可见度剪枝
python train.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/StandUp_V9_3_NoisePrune_White_NoVisPrune \
--iterations 30000 \
--white_background \
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
--visibility_prune_min_count 0 \
--visibility_prune_interval 200 \
--visibility_prune_start_iter 800

无后处理
这一项不需要重新训练，直接用“全部开启”的模型渲染时关掉后处理就行。也就是说，训练命令还是基线那一版，渲染时改成
python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/StandUp_V9_3_NoisePrune_White \
--iteration 30000 \
--white_background \
--num_output_frames 420 \
--video_fps 30 \
--lock_camera \
--deform_time_samples 2 \
--deform_time_window 0.001 \
--deform_time_sigma 0.0005 \
--post_smooth_mode none