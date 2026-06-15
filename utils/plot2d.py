from matplotlib import pyplot as plt
from shapely import bounds
import torch
from utils.constants import BASE_DIR


def identify_samples_which_satisfy_constraints(X, constraints):
    """
    Takes in values (a1, ..., ak, o) and returns (a1, ..., ak, o)
    True/False values, where o is the number of outputs.
    """
    successful = torch.ones(X.shape).to(X)
    for model_index in range(X.shape[-1]):
        these_X = X[..., model_index]
        direction, value = constraints[model_index]
        successful[..., model_index] = (
            these_X < value if direction == "lt" else these_X > value
        )
    return successful

def plot_results(X, Y, task_func, bounds, constraints, figpath):
    """
    Plots the results of the 2D CAS-loop, visualizing the feasible region 
    and the points chosen by the algorithm.
    """
    N1, N2 = 100, 100
    device = X.device

    Xplt, Yplt = torch.meshgrid(
        torch.linspace(bounds[0, 0], bounds[1, 0], N1, device=device, dtype=torch.double),
        torch.linspace(bounds[0, 1], bounds[1, 1], N2, device=device, dtype=torch.double), indexing="ij"
    )
    xplt = torch.stack(
        (
            torch.reshape(Xplt, (Xplt.shape[0] * Xplt.shape[1],)),
            torch.reshape(Yplt, (Yplt.shape[0] * Yplt.shape[1],)),
        ),
        dim=1,
    )
    yplt = task_func(xplt)
    Zplt = torch.reshape(yplt[:, 0], (N1, N2))  # Since f1(x) = f2(x)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    h1 = ax.contourf(Xplt.cpu().numpy(), Yplt.cpu().numpy(), Zplt.cpu().numpy(), 20, cmap="Blues", alpha=0.6)
    fig.colorbar(h1)
    
    # Dynamically compute and plot the ground truth feasible region & boundary
    yplt_grid = torch.reshape(yplt, (N1, N2, -1))
    gt_feasible_mask = identify_samples_which_satisfy_constraints(yplt_grid, constraints)
    gt_feasible_region = gt_feasible_mask.prod(dim=-1).to(torch.double)
    ax.contourf(Xplt.cpu().numpy(), Yplt.cpu().numpy(), gt_feasible_region.cpu().numpy(), levels=[0.5, 1.5], colors='lightgreen', alpha=0.3)
    ax.contour(Xplt.cpu().numpy(), Yplt.cpu().numpy(), gt_feasible_region.cpu().numpy(), levels=[0.5], colors='darkgreen', linestyles='-', linewidths=2)

    feasible_inds = (
        identify_samples_which_satisfy_constraints(Y, constraints)
        .prod(dim=-1)
        .to(torch.bool)
    )
    ax.plot(X[feasible_inds, 0].cpu(), X[feasible_inds, 1].cpu(), "sg", label="Feasible")
    ax.plot(
        X[~feasible_inds, 0].cpu(), X[~feasible_inds, 1].cpu(), "sr", label="Infeasible"
    )

    ax.legend(loc=[0.7, 0.05])
    ax.set_title("$f_1(x)$")  # Recall that f1(x) = f2(x)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", "box")
    ax.set_xlim([bounds[0, 0].item(), bounds[1, 0].item()])
    ax.set_ylim([bounds[0, 1].item(), bounds[1, 1].item()])
    
    plt.savefig(figpath, dpi=300, bbox_inches='tight')
    


# def plot_results(train_X, train_Y, task_func, bounds, constraints):
#     """
#     Plots the results of the 2D CAS-loop, visualizing the feasible region 
#     and the points chosen by the algorithm.
#     """
#     # 1. Create a dense grid to map the background contour
#     x1 = torch.linspace(bounds[0, 0], bounds[1, 0], 100, dtype=torch.double)
#     x2 = torch.linspace(bounds[0, 1], bounds[1, 1], 100, dtype=torch.double)
#     X1, X2 = torch.meshgrid(x1, x2, indexing="ij")
#     Xplt = torch.stack([X1, X2], dim=-1)
    
#     # 2. Evaluate the true function to plot the contour surface
#     Yplt = task_func(Xplt)
#     Zplt = Yplt[..., 0]  # Plotting the first objective for the background map
    
#     fig, ax = plt.subplots(figsize=(8, 6))
#     h1 = ax.contourf(X1.numpy(), X2.numpy(), Zplt.numpy(), 20, cmap="Blues", alpha=0.6)
#     fig.colorbar(h1)
    
#     # 3. Identify which sampled points satisfied all constraints
#     feasible_inds = (
#         identify_samples_which_satisfy_constraints(train_Y, constraints)
#        .prod(dim=-1)
#        .to(torch.bool)
#     )
    
#     # 4. Plot feasible (green) and infeasible (red) points
#     ax.plot(train_X[feasible_inds, 0].numpy(), train_X[feasible_inds, 1].numpy(), "sg", label="Feasible")
#     ax.plot(train_X[~feasible_inds, 0].numpy(), train_X[~feasible_inds, 1].numpy(), "sr", label="Infeasible")
    
#     # 5. Plot the ground truth feasible region
#     # Evaluate the constraints on the dense grid to get a boolean mask for the feasible region
#     ground_truth_feasible_mask_per_objective = identify_samples_which_satisfy_constraints(Yplt, constraints)
#     ground_truth_feasible_region = ground_truth_feasible_mask_per_objective.prod(dim=-1).to(torch.bool)

#     # Plot the ground truth feasible region as a filled contour
#     ax.contourf(X1.numpy(), X2.numpy(), ground_truth_feasible_region.numpy(), levels=[0.5, 1.5], colors='lightgreen', alpha=0.3)
#     # Plot the boundary of the ground truth feasible region
#     ax.contour(X1.numpy(), X2.numpy(), ground_truth_feasible_region.numpy(), levels=[0.5], colors='darkgreen', linestyles='-', linewidths=2)

#     # Update legend to include ground truth plots
#     legend_handles, legend_labels = ax.get_legend_handles_labels()
#     legend_handles.append(Patch(facecolor='lightgreen', alpha=0.3, label='Ground Truth Feasible Region'))
#     legend_handles.append(plt.Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=2, label='Ground Truth Feasible Boundary'))
#     ax.legend(handles=legend_handles, loc="lower right")

#     ax.set_title("Constraint Active Search: Feasible vs Infeasible Samples")
#     ax.set_xlabel("$x_1$")
#     ax.set_ylabel("$x_2$")
#     ax.set_aspect("equal", "box")
#     ax.set_xlim([bounds[0, 0].item(), bounds[1, 0].item()])
#     ax.set_ylim([bounds[0, 1].item(), bounds[1, 1].item()])
#     plt.show()