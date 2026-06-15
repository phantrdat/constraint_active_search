from networkx import radius
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.models.transforms.input import Normalize
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.pareto import is_non_dominated

def identify_feasible(Y, constraints):
    """Returns a boolean mask of samples satisfying all constraints."""
    feasible = torch.ones(Y.shape[:-1], dtype=torch.bool, device=Y.device)
    for i, (direction, threshold) in enumerate(constraints):
        if direction == "lt":
            feasible &= Y[..., i] <= threshold
        else:
            feasible &= Y[..., i] >= threshold
    return feasible


def compute_metrics(train_X, train_Y, S_grid_X, constraints, ref_point, radius, bounds=None):
    """
    Computes the 4 benchmarking metrics at the current iteration.
    """
    # 1. Positive Samples (Cumulative feasible observations)
    feas_mask = identify_feasible(train_Y, constraints)
    pos_samples = feas_mask.sum().item()
    
    # Calculate distances in PARAMETER SPACE (train_X) as per Malkomes et al. 2021
    # Fill distance and coverage recall measure the representation of the feasible region S in parameter space.
    if len(S_grid_X) > 0 and len(train_X) > 0:
        # Filter train_X to only include points that are actually feasible
        observed_feasible_X = train_X[feas_mask]
        
        if len(observed_feasible_X) > 0:
            # Normalize coordinates to [0, 1] if bounds are provided.
            # This ensures radius=0.1 consistently means "within 10% of the range"
            # and prevents cov_recall from being 0.0 due to high-dimensional distance scaling.
            if bounds is not None:
                S_ref = (S_grid_X - bounds[0]) / (bounds[1] - bounds[0] + 1e-9)
                obs_ref = (observed_feasible_X - bounds[0]) / (bounds[1] - bounds[0] + 1e-9)
            else:
                S_ref, obs_ref = S_grid_X, observed_feasible_X

            dists = torch.cdist(S_ref, obs_ref)
            min_dists = dists.min(dim=1).values

            # 2. Fill Distance (normalized if bounds provided)
            fill_dist = min_dists.max().item()

            # 3. Coverage Recall
            covered_count = (min_dists <= radius).sum().item()
            cov_recall = covered_count / len(S_grid_X)
        else:
            fill_dist = float("nan")
            cov_recall = 0.0
    else:
        fill_dist = float("nan")
        cov_recall = 0.0

    # 4. Hypervolume of the feasible subset
    feas_Y = train_Y[feas_mask]
    if len(feas_Y) > 0:
        pareto_Y = feas_Y[is_non_dominated(feas_Y)]

        hv = Hypervolume(ref_point)
        hv_val = hv.compute(pareto_Y)
    else:
        hv_val = 0.0
    
    # print("num observed feasible =", len(observed_feasible_X))

    # print(
    #     "min/mean/max dist:",
    #     min_dists.min().item(),
    #     min_dists.mean().item(),
    #     min_dists.max().item())

    # print("radius =", radius)
    
    return pos_samples, fill_dist, hv_val, cov_recall


def initialize_models(train_X, train_Y, bounds):
    models = []
    input_transform = Normalize(d=train_X.shape[-1], bounds=bounds)
    for i in range(train_Y.shape[-1]):
        train_y_task = train_Y[..., i : i + 1]
        models.append(SingleTaskGP(train_X, train_y_task, input_transform=input_transform))
    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    return mll, model_list