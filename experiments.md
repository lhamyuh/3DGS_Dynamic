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