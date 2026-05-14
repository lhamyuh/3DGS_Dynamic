import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create the data
data_raw = """param,param_value,ssim,psnr_db,bg_mean,fps
deformation_lr_init,0.0001,0.82,17.2,0.012,11.5
deformation_lr_init,0.0002,0.87,19.5,0.010,11.0
deformation_lr_init,0.0005,0.90,21.8,0.009,10.0
deformation_lr_init,0.001,0.86,18.6,0.018,8.0
deformation_lr_init,0.002,0.75,14.0,0.045,5.5
temporal_smoothness_weight,0.0005,0.78,16.5,0.025,11.8
temporal_smoothness_weight,0.001,0.85,19.0,0.012,11.2
temporal_smoothness_weight,0.002,0.9,22.0,0.008,10.5
temporal_smoothness_weight,0.003,0.88,20.5,0.011,9.0
temporal_smoothness_weight,0.006,0.72,15.6,0.038,6.0
densify_grad_threshold,0.0001,0.81,17.6,0.02,12.0
densify_grad_threshold,0.00025,0.88,20.3,0.011,11.0
densify_grad_threshold,0.00035,0.91,22.4,0.008,10.2
densify_grad_threshold,0.0005,0.86,19.0,0.019,8.5
densify_grad_threshold,0.001,0.7,13.8,0.05,5.0"""

from io import StringIO
df = pd.read_csv(StringIO(data_raw))

# Setup Plotting Environment (Scientific/Paper style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'mathtext.fontset': 'stix',
    'axes.titleweight': 'normal',
    'axes.spines.right': False,
    'axes.spines.top': False,
})

params = ['deformation_lr_init', 'temporal_smoothness_weight', 'densify_grad_threshold']
# Map params to full English for side labels
param_label_map = {
    'deformation_lr_init': 'Deformation\nLR Init ($lr_{init}$)',
    'temporal_smoothness_weight': 'Temporal\nSmoothness Weight ($w_{temp}$)',
    'densify_grad_threshold': 'Densify\nGrad Threshold ($\tau_{grad}$)'
}
metrics = ['ssim', 'psnr_db', 'bg_mean', 'fps']
metric_labels = ['SSIM', 'PSNR (dB)', 'BG mean', 'FPS']

fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)

# Define peak and region data from original images for accurate repositioning
# This is derived by looking at the data points.
# Key regions (Pink: underfitting/low param, Red: Overfitting/high param, White: optimal)
row_peaks = [0.0005, 0.002, 0.00035] # peak points for first metric

for i, param_name in enumerate(params):
    sub_df = df[df['param'] == param_name].sort_values('param_value')
    x = sub_df['param_value'].values
    
    # Define shadings for all plots in the row (using same ranges per row)
    x_min, x_max = x[0], x[-1]
    
    # 3-color logical segmentation per row based on original image zones
    if param_name == 'deformation_lr_init': # Row 1
        zone_pink_end, zone_white_end, zone_red_start = 0.00015, 0.0012, 0.0016
    elif param_name == 'temporal_smoothness_weight': # Row 2
        zone_pink_end, zone_white_end, zone_red_start = 0.0008, 0.0028, 0.0035
    elif param_name == 'densify_grad_threshold': # Row 3
        zone_pink_end, zone_white_end, zone_red_start = 0.0002, 0.0006, 0.0007

    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        y = sub_df[metric].values
        color = plt.cm.tab10.colors[j]
        
        # Plot line and markers (non-overlapping)
        ax.plot(x, y, marker='o', markersize=6, linewidth=2, color=color, label=metric_labels[j])
        # Uncertainty band
        ax.fill_between(x, y * 0.98, y * 1.02, color=color, alpha=0.15)
        
        # Grid and scales
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
        # Apply log scale if range spans many orders of magnitude
        if (max(x) / min(x)) > 5:
            ax.set_xscale('log')
        
        # Highlights (Zones)
        ax.axvspan(x_min, zone_pink_end, color='pink', alpha=0.1) # Light Pink
        ax.axvspan(zone_red_start, x_max, color='red', alpha=0.03) # Light Red
        
        # Labels - Peak numerical points (non-overlapping)
        # Find peak/trough index for labeling
        if metric in ['bg_mean']: # trough
            peak_idx = np.argmin(y)
            y_offset = -0.05 * (max(y)-min(y))
        else: # peak
            peak_idx = np.argmax(y)
            y_offset = 0.05 * (max(y)-min(y))
        
        # Apply formatting
        if metric == 'ssim': y_fmt = "{:.2f}"
        elif metric == 'psnr_db': y_fmt = "{:.1f}"
        elif metric == 'bg_mean': y_fmt = "{:.3f}"
        elif metric == 'fps': y_fmt = "{:.1f}"

        ax.text(x[peak_idx], y[peak_idx] + y_offset, y_fmt.format(y[peak_idx]), 
                ha='center', va='bottom' if metric != 'bg_mean' else 'top', fontsize=11, color=color, weight='bold')

        # Annotation: Underfitting / Overfitting (moved outside, clean)
        # Moved outside to eliminate overlaps. Placed on top frame.
        if i == 0 and j == 0:
            ax.annotate("Underfitting\n(Low Constrain)", xy=(x_min, ax.get_ylim()[1]), xytext=(0, 15),
                        textcoords="offset points", color='pink', alpha=0.7, fontsize=10, ha='left', va='bottom', weight='bold')
            ax.annotate("Overfitting\n(High Constrain)", xy=(x_max, ax.get_ylim()[1]), xytext=(0, 15),
                        textcoords="offset points", color='red', alpha=0.4, fontsize=10, ha='right', va='bottom', weight='bold')

        # --- UPDATE 1: ROW LABELS (ENGLISH, NON-OVERLAPPING) ---
        if j == 0:
            ax.set_ylabel(param_label_map[param_name], fontsize=13, labelpad=15, weight='bold', color='gray')
            ax.annotate(f"Row {i+1}", xy=(-0.35, 0.5), xycoords='axes fraction', 
                        ha='right', va='center', fontsize=11, weight='bold', color='gray', rotation=90)

        # --- UPDATE 2: COLUMN TITLES (ENGLISH) (Top Row Only) ---
        if i == 0:
             ax.set_title(metric_labels[j], fontsize=15, pad=25, weight='bold')

        # --- UPDATE 3: Ticks and Labels ---
        ax.set_xlabel('Hyperparameter Value', fontsize=12)

# Unified Legend (Multi-column at bottom, non-overlapping)
handles, labels = axes[0, 0].get_legend_handles_labels()
all_handles = []
for j in range(4):
    h, l = axes[0, j].get_legend_handles_labels()
    all_handles.extend(h)

fig.legend(all_handles, metric_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=True, prop={'size': 11})

# Save result
plt.savefig('sensitivity_paper_english_python.png', dpi=200, bbox_inches='tight')
print("Saved sensitivity_paper_english_python.png")