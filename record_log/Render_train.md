none（弱平滑，保细节）

python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_2_NoisePrune \
--iteration 30000 \
--num_output_frames 420 \
--video_fps 30 \
--lock_camera \
--deform_time_samples 2 \
--deform_time_window 0.001 \
--deform_time_sigma 0.0005 \
--post_smooth_mode none


EMA
python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_2_NoisePrune \
--iteration 30000 \
--num_output_frames 420 \
--video_fps 30 \
--lock_camera \
--deform_time_samples 2 \
--deform_time_window 0.001 \
--deform_time_sigma 0.0005 \
--post_smooth_mode ema \
--ema_alpha 0.7

median

python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_2_NoisePrune \
--iteration 30000 \
--num_output_frames 420 \
--video_fps 30 \
--lock_camera \
--deform_time_samples 2 \
--deform_time_window 0.001 \
--deform_time_sigma 0.0005 \
--post_smooth_mode median \
--smooth_radius 2

background cleanup

python render_4d.py \
-s /autodl-fs/data/dataset_dynamic/data/standup \
-m output/Standup_V9_3_2_NoisePrune \
--iteration 30000 \
--num_output_frames 420 \
--video_fps 30 \
--lock_camera \
--deform_time_samples 2 \
--deform_time_window 0.001 \
--deform_time_sigma 0.0005 \
--post_smooth_mode none \
--bg_cleanup \
--bg_cleanup_use_alpha \
--bg_cleanup_alpha_threshold 0.5 \
--bg_cleanup_blend 1.0