import json
import os
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from tqdm import tqdm

import time
# Import the baselines you previously saved
from baselines.eci import ExpectedCoverageImprovement
from baselines.moc_cas import MOCCAS
from baselines.seq import SingleObjectiveStraddle
from baselines.straddle import Straddle
from baselines.lookahead_cas import LookAheadCAS
from baselines.volume_cas import VolumeCAS
from utils.plot2d import plot_results # This will be disabled for high-dim
from utils.benchmark_helpers import compute_metrics, initialize_models, identify_feasible
from utils.plot_metrics import plot_metrics
from utils.constants import BASE_DIR
from exps.synthetic import get_experiment_setup as get_synthetic_setup

plt.rcParams.update({'text.usetex': False})
# ==========================================
# 4. Main Benchmarking Loop
# ==========================================
def run_benchmarks(baselines=["ECI"], num_trials=3, num_iterations=20, n_init=5, punchout_radius=0.1, experiment_name="toy_2d"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fallback to synthetic for original functionality
    
    task_func, bounds, constraints = get_synthetic_setup(experiment_name)
    dim = bounds.shape[1]
    
    # Move bounds to device
    bounds = bounds.to(device)
    ref_point = torch.zeros(len(constraints), device=device, dtype=torch.double) # Dynamic size based on constraints

    # For higher dimensions, a dense grid is computationally infeasible.
    # We fallback to a large Monte Carlo sample to approximate the feasible region S.
    if dim <= 2:
        grid_x = torch.linspace(bounds[0, 0], bounds[1, 0], 100, device=device, dtype=torch.double)
        grid_y = torch.linspace(bounds[0, 1], bounds[1, 1], 100, device=device, dtype=torch.double)
        X1, X2 = torch.meshgrid(grid_x, grid_y, indexing="ij")
        X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    else:
        # High-dimensional approximation using uniform random sampling
        num_samples = 50000 
        X_grid = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(num_samples, dim, device=device, dtype=torch.double)

    # Evaluate the grid in objective space and filter for feasible points
    Y_grid = task_func(X_grid, noise_std=0.)
    
    # for z in range(Y_grid.shape[1]):
    #     print(f"Dim {z}, Min {torch.min(Y_grid[:, z]):.4f} | Max {torch.max(Y_grid[:, z]):.4f} | Mean {torch.mean(Y_grid[:, z]):.4f}")
    
    feasible_mask = identify_feasible(Y_grid, constraints)
    S_grid_X = X_grid[feasible_mask]
    
    print(f"Experiment: {experiment_name} | Dim: {dim} | S_size: {len(S_grid_X)} | Feasible Fraction: {feasible_mask.float().mean().item():.4f}")
    
    
    if len(S_grid_X) == 0:
        print(f"WARNING: No feasible points found in ground-truth grid for '{experiment_name}'. "
              "Try increasing num_samples for the Monte Carlo approximation.")

    dim = bounds.shape[1]

    
    metrics_history = { b: {"pos_samples": [], 
                           "fill_dist": [], 
                           "hv": [], 
                           "cov_recall": [], 
                           } for b in baselines}
    metrics_history["experiment_name"] = experiment_name
    for trial in range(num_trials):
        
        for baseline_id, baseline in enumerate(baselines):
            
            baseline_dir = f"{BASE_DIR}/numerical_results/{baseline}_{experiment_name}"
            os.makedirs(baseline_dir, exist_ok=True)
            
            # Aggregated result file for the entire baseline
            baseline_summary_file = f"{baseline_dir}/{experiment_name}_{baseline}_{num_trials}trials_{num_iterations}iterations_{n_init}init_summary.pt"

            # Define the specific file for this baseline, experiment, and trial
            trial_file = f"{baseline_dir}/{experiment_name}_{baseline}_trial_{trial+1}_{num_iterations}iterations_{n_init}init.pt"
            
            pbar = tqdm(enumerate(range(num_iterations)), total=num_iterations, leave=False)
            
            torch.manual_seed(0)
            if os.path.exists(trial_file):
                pbar.set_description(f"Exp {trial+1}/{num_trials} | {baseline} ({baseline_id + 1}/{len(baselines)})| Loading existing results...")
                trial_metrics = torch.load(trial_file)
                for metric in trial_metrics:
                    metrics_history[baseline][metric].append(trial_metrics[metric])
                
                time.sleep(0.5) # Pause for 0.5 seconds before closing
                pbar.close()
                continue

            
            # Shared initial data for fairness via deterministic seeding per trial
            torch.manual_seed(trial)

            train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, dim, device=device, dtype=torch.double)
            train_Y = task_func(train_X)

            # Temporary storage for the current trial
            trial_metrics = {"pos_samples": [], "fill_dist": [], "hv": [], "cov_recall": []}
            
            for i, _ in pbar:
                mll, model = initialize_models(train_X, train_Y, bounds)
                fit_gpytorch_mll(mll)

                if baseline == "ECI":
                    acqf = ExpectedCoverageImprovement(
                        model=model,
                        constraints=constraints,
                        punchout_radius=punchout_radius,
                        bounds=bounds,
                        train_inputs=train_X
                    )
                elif baseline == "MOC-CAS":
                    acqf = MOCCAS(
                        model=model,
                        constraints=constraints,
                        previous_y=train_Y,
                        radius=punchout_radius,
                        beta=2.0
                    )
                elif baseline == "Straddle":
                    acqf = Straddle(
                        model=model,
                        constraints=constraints,
                        beta=1.96**2,
                        lambda_penalty=1.0
                    )
                elif baseline == "VolumeCAS":
                    acqf = VolumeCAS(
                        model=model,
                        constraints=constraints,
                        beta=1.96**2,
                        gamma=2.0
                    )
                elif baseline == "LookAheadCAS":
                    grid_Z = draw_sobol_samples(bounds, n=5000,q=1).squeeze(1)
                    feasible_mask = identify_feasible(
                        train_Y,
                        constraints
                    )

                    train_X_feasible = train_X[feasible_mask]
                    if train_X_feasible.shape[0] == 0:
                        train_X_feasible = train_X[:1]
                    acqf = LookAheadCAS(
                        model=model,
                        constraints=constraints,
                        train_X=train_X_feasible,
                        grid_Z=grid_Z,
                        candidate_set=grid_Z,
                        radius=punchout_radius,
                        k=3,
                        beta=1.96**2,
                        gamma=2.0
                    )
                elif baseline == "SEQ":
                    # 1. Determine which objective to focus on based on the current iteration
                    num_objectives = len(constraints)
                    phase_length = max(1, num_iterations // num_objectives)
                    
                    # Calculate current phase (capped at the last objective)
                    current_obj_idx = min(i // phase_length, num_objectives - 1)
                    
                    # Get the threshold for the active objective
                    direction, threshold = constraints[current_obj_idx]
                    
                    # 2. Extract the specific SingleTaskGP for the current objective
                    single_model = model.models[current_obj_idx]
                    
                    # 3. Instantiate the single-objective acquisition function
                    acqf = SingleObjectiveStraddle(
                        model=single_model,
                        threshold=threshold,
                        beta=4.0
                    )
                if baseline == "LookAheadCAS":
                    X_cand = draw_sobol_samples(bounds, n=500, q=1)
                    candidate_idx = acqf(X_cand).argmax()
                    candidates = X_cand[candidate_idx]
                else:
                # Optimize Acquisition Function
                    candidates, _ = optimize_acqf(
                            acq_function=acqf,
                            bounds=bounds,
                            q=1,
                            num_restarts=5,
                            raw_samples=256,
                        )

            
                new_X = candidates.detach()
                new_Y = task_func(new_X)

                train_X = torch.cat([train_X, new_X], dim=0)
                train_Y = torch.cat([train_Y, new_Y], dim=0)

                # Evaluate Metrics
                pos, fd, hv, cr = compute_metrics(
                    train_X, 
                    train_Y, 
                    S_grid_X, 
                    constraints, 
                    ref_point, 
                    punchout_radius,
                    bounds=bounds, verbose=args.v
                )
                trial_metrics["pos_samples"].append(pos)
                trial_metrics["fill_dist"].append(fd)
                trial_metrics["hv"].append(hv)
                trial_metrics["cov_recall"].append(cr)
                

                res = f"Iter {i+1:02d} | Pos: {pos:02d} | Fill Dist: {fd:.3f} | HV: {hv:.3f} | Cov Recall: {cr:.3f}"
                pbar.set_description(f"Exp {trial+1}/{num_trials} | {baseline} ({baseline_id + 1}/{len(baselines)})| {res}")                
            
            pbar.close()
            # Store trial progression
            for metric in trial_metrics:
                metrics_history[baseline][metric].append(trial_metrics[metric])
            
            # Save individual trial results to disk
            torch.save(trial_metrics, trial_file)
            
            if dim == 2:
                figpath=f"{BASE_DIR}/figures/{baseline}_{experiment_name}/{experiment_name}_{baseline}_{experiment_name}_trial_{trial+1}_results.png" 
                if not os.path.exists(os.path.dirname(figpath)):
                    os.makedirs(os.path.dirname(figpath), exist_ok=True)
                # Generate and save a 2D plot for the current baseline run (only for 2D synthetic)
                plot_results(
                    X=train_X,
                    Y=train_Y,
                    task_func=task_func,
                    bounds=bounds,
                    constraints=constraints,
                    figpath=figpath
                )

        # Save the aggregated results for this baseline
        torch.save(metrics_history[baseline], baseline_summary_file)

    # Save the metrics history for later analysis
    f_name = f"{BASE_DIR}/numerical_results/{experiment_name}_{num_trials}trials_{num_iterations}iterations_{n_init}init.pt"
    torch.save(metrics_history, f_name)
    return metrics_history



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action='store_true', default=False, help="Whether to print verbose output")
    args = parser.parse_args()
    running_settings = json.load(open(f"{BASE_DIR}/running_settings.json"))
    tasks = running_settings["tasks"]
    baselines = running_settings["baselines"]
    
    print("Total tasks to run:", len(tasks))
    for t in tasks:
        history = run_benchmarks(baselines=baselines, num_trials=10, num_iterations=t["num_iterations"], n_init=t["n_init"], punchout_radius=t["punchout_radius"], experiment_name=t["function_name"])
        # plot_metrics(history)