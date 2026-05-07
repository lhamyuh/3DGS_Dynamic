Tensorboard可视化监控：
# 在 D:\3DGS_Project\python\gaussian-splatting 目录下运行 
tensorboard --logdir=E:/3dgsData/output/counter_reproduce
http://localhost:6006

含有sparse的源数据集处理命令：
(3dgs_env) D:\3DGS_Project\python\gaussian-splatting>
python train.py -s D:/3DGS_Project/data/mileleap-mipnerf360/garden -m output/garden_reproduce --eval -r 4

含有json的合成源数据集处理命令：
python train.py -s D:/3DGS_Project/data/nerf_synthetic/mic --eval -m output/mic_reproduce

远程链接vpn
ssh -R 23333:127.0.0.1:10808 root@connect.westd.seetacloud.com -p 22575

训练
python train.py \
    -s ../dataset_dynamic/data/standup \
    -m output/Standup_V8_Smooth \
    --iterations 30000 \
    --deformation_lr_init 0.001 \
    --temporal_smoothness_weight 0.003 \
    --temporal_smoothness_start_iter 3000 \
    --temporal_smoothness_epsilon 0.01 \
--prune_opacity_threshold 0.005

生成视频
固定视角
python render_4d.py \
  -s /autodl-fs/data/dataset_dynamic/data/standup \
  -m output/Standup_V9_1_SmoothDyn \
  --iteration 30000 \
  --num_output_frames 420 \
  --video_fps 30 \
  --lock_camera \
  --deform_time_samples 2 \
  --deform_time_window 0.001 \
  --deform_time_sigma 0.0005 \
  --post_smooth_mode ema \
  --ema_alpha 0.7


自由视角

python render_4d.py \
  -s /autodl-fs/data/dataset_dynamic/data/standup \
  -m output/Standup_V9_1_SmoothDyn \
  --iteration 30000 \
  --num_output_frames 840 \
  --video_fps 24 \
  --camera_time_samples 1 \
  --camera_time_window 0.0 \
  --camera_time_sigma 0.0 \
  --deform_time_samples 1 \
  --deform_time_window 0.0 \
  --deform_time_sigma 0.0 \
  --post_smooth_mode none


查看已训练好的（本地）
静态：
.\SIBR_gaussianViewer_app.exe -m E:\3dgsData\output\Standup_V9_1_SmoothDyn -s E:\3dgsData\output\Standup_V9_1_SmoothDyn --iteration 30000

动态：
.\SIBR_gaussianViewer_app.exe -m E:\3dgsData\output\Standup_V9_1_SmoothDyn -s E:\3dgsData\output\Standup_V9_1_SmoothDyn --iteration 30000 --sequence_dir E:\3dgsData\output\Standup_V9_1_SmoothDyn\ply_sequence --sequence_fps 24 --sequence_loop

查看正在训练云端

导出连续ply文件
python export_4d_ply.py \
  -s /autodl-fs/data/dataset_dynamic/data/standup \
  -m output/Standup_V9_1_SmoothDyn \
  --iteration 30000 \
  --num_output_frames 840 \
  --deform_time_samples 1 \
  --deform_time_window 0.0 \
  --deform_time_sigma 0.0 \
  --sequence_dir output/Standup_V9_1_SmoothDyn/ply_sequence \
  --file_prefix frame
