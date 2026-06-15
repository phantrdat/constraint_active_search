import os

from matplotlib import pyplot as plt
import numpy as np
from tueplots import bundles, figsizes, axes

from utils.constants import BASE_DIR

def plot_metrics(metrics_history):
    # 1. Update rcParams using the tueplots bundle for ICML 2024. 
    # (usetex=False avoids needing a full local LaTeX installation to render)
    plt.rcParams.update(bundles.aaai2024())
    
    # 2. Adjust the figure size to span the full text width with a 2x2 grid
    plt.rcParams.update(figsizes.icml2024_full(nrows=2, ncols=2))
    
    # 3. Apply clean, publication-ready axis line widths and tick marks
    plt.rcParams.update(axes.lines())

    # Create the subplots (tueplots handles the figsize automatically now)
    fig, axes_arr = plt.subplots(2, 2)
    fig.suptitle("Constraint Active Search Benchmarking")
    
    baseline_markers = {
        "ECI": "o",
        "MOC-CAS": "s",
        "VolumeCAS": "^",
        "Straddle": "D",
        "LookaheadCAS": "P",
    }
    
    metrics = [
        ("pos_samples", "Cumulative Feasible Samples", axes_arr[0, 0]),
        ("fill_dist", "Fill Distance", axes_arr[0, 1]),
        ("hv", "Hypervolume", axes_arr [1, 0]),
        ("cov_recall", "Coverage Recall", axes_arr[1, 1])
    ]
    
    for metric_key, title, ax in metrics:
        for baseline, data in metrics_history.items():
            if baseline == "experiment_name":
                continue
            # Convert list of lists to a numpy array for fast mean/std calculation
            # Shape will be: (num_trials, num_iterations)
            data_np = np.array(data[metric_key])
            
            mean_vals = np.mean(data_np, axis=0)
            std_vals = np.std(data_np, axis=0)
            
            # X-axis array for plotting
            iterations = np.arange(1, len(mean_vals) + 1)
            
            # Plot the mean line
            marker_style = baseline_markers.get(baseline, "x") # Use 'x' as a default
            line, = ax.plot(iterations, mean_vals, label=baseline, marker=marker_style, markersize=4, markevery=2)
            
            # Draw the shaded standard deviation region
            ax.fill_between(
                iterations, 
                mean_vals - std_vals, 
                mean_vals + std_vals, 
                color=line.get_color(), 
                alpha=0.2,
                edgecolor="none"
            )
            
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first plot to save space in the paper
        if metric_key == "pos_samples":
            ax.legend(loc="upper left")

    
    fig_path = os.path.join(BASE_DIR, f"figures/benchmark_metrics_{metrics_history['experiment_name']}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')