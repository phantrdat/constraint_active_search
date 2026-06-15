from datetime import datetime

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood

# Import the baselines and experiment setups
from baselines.volume_cas import VolumeCAS
from exps.synthetic import get_experiment_setup
from utils.benchmark_helpers import compute_metrics, initialize_models
from utils.plot_metrics import plot_metrics


def run_gamma_ablation(
    gamma_values,
    num_trials=3,
    num_iterations=20,
    n_init=5,
    punchout_radius=0.1,
    experiment_name="toy_2d",
):
    task_func, bounds, constraints = get_experiment_setup(experiment_name)
    dim = bounds.shape[1]
    ref_point = torch.tensor([0.0, 0.0], dtype=torch.double)  # For Hypervolume

    # Create a dense grid to represent the true Satisfactory Region (S)
    grid_x = torch.linspace(bounds[0, 0], bounds[1, 0], 50, dtype=torch.double)
    grid_y = torch.linspace(0, bounds[1, 1], 50, dtype=torch.double)
    X1, X2 = torch.meshgrid(grid_x, grid_y, indexing="ij")
    X_grid = torch.stack([X1.flatten(), X2.flatten()], dim=-1)

    S_grid = X_grid

    baselines = [f"VolumeCAS (gamma={g:.1f})" for g in gamma_values]
    metrics_history = {
        b: {
            "pos_samples": [],
            "fill_dist": [],
            "hv": [],
            "cov_recall": [],
        }
        for b in baselines
    }
    metrics_history["experiment_name"] = f"{experiment_name}_gamma_ablation"

    for trial in range(num_trials):
        print(f"\n==========================================")
        print(f"               TRIAL {trial + 1}/{num_trials}")
        print(f"==========================================")

        # Shared initial data for fairness
        initial_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
            n_init, dim, dtype=torch.double
        )
        initial_Y = task_func(initial_X)

        for i, gamma in enumerate(gamma_values):
            baseline_name = baselines[i]
            print(f"\n--- Running Ablation: {baseline_name} ---")
            train_X = initial_X.clone()
            train_Y = initial_Y.clone()

            trial_metrics = {
                "pos_samples": [],
                "fill_dist": [],
                "hv": [],
                "cov_recall": [],
            }

            for i in range(num_iterations):
                mll, model = initialize_models(train_X, train_Y, bounds)
                fit_gpytorch_mll(mll)

                acqf = VolumeCAS(
                    model=model, constraints=constraints, beta=1.96**2, gamma=gamma
                )

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

                pos, fd, hv, cr = compute_metrics(
                    train_X, 
                    train_Y, 
                    S_grid, 
                    constraints, 
                    ref_point, 
                    punchout_radius,
                    bounds=bounds
                )
                trial_metrics["pos_samples"].append(pos)
                trial_metrics["fill_dist"].append(fd)
                trial_metrics["hv"].append(hv)
                trial_metrics["cov_recall"].append(cr)

            for metric in trial_metrics:
                metrics_history[baseline_name][metric].append(trial_metrics[metric])

    f_name = f"numerical_results/gamma_ablation_{experiment_name}_{num_trials}trials_{num_iterations}iterations_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"
    torch.save(metrics_history, f_name)
    return metrics_history


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    # Define the gamma values for the ablation study
    gamma_values_to_test = [2, 1.5, 1, 4.0]

    # Run the ablation study
    history = run_gamma_ablation(
        gamma_values=gamma_values_to_test,
        num_trials=3,
        num_iterations=25,
        n_init=5,
        punchout_radius=0.1,
        experiment_name="toy_2d",
    )

    # Plot the comparative metrics
    plot_metrics(history)
