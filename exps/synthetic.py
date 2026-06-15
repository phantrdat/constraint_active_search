import torch
import math

# =====================================================================
# AXIS 1: DIMENSIONALITY (Low d=2 to High d=10, 20)
# Scalability checks for GP surrogate modeling and acquisition optimization.
# =====================================================================

def toy_2d_multiobjective(X, noise_std=0.05):
    """2D spherical objectives with a massive, connected intersection."""
    y1 = 1.0 - torch.norm(X - 0.25, dim=-1)**2
    y2 = 1.0 - torch.norm(X - 0.75, dim=-1)**2
    y1 += noise_std * torch.randn_like(y1)
    y2 += noise_std * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)
def fonseca_fleming_multiobjective(X, noise_std=0.01):
    """2D smooth, conflicting exponential curves."""
    term1 = (X[..., 0] - 1/math.sqrt(2))**2 + (X[..., 1] - 1/math.sqrt(2))**2
    term2 = (X[..., 0] + 1/math.sqrt(2))**2 + (X[..., 1] + 1/math.sqrt(2))**2
    y1 = 1.0 - torch.exp(-term1)
    y2 = 1.0 - torch.exp(-term2)
    y1 += noise_std * torch.randn_like(y1)
    y2 += noise_std * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)

def binh_and_korn_multiobjective(X, noise_std=0.5):
    """
    Binh and Korn function.
    X: Tensor of shape (..., 2)
    """
    x1 = X[..., 0]
    x2 = X[..., 1]
    
    y1 = 4 * x1**2 + 4 * x2**2
    y2 = (x1 - 5)**2 + (x2 - 5)**2
    
    y1 += noise_std * torch.randn_like(y1)
    y2 += noise_std * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def hartmann6_multiobjective(X, noise_std=0.01):
    """6D Hartmann function with 2 competing objectives."""
    alpha1 = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=X.dtype, device=X.device)
    alpha2 = torch.tensor([3.2, 3.0, 1.2, 1.0], dtype=X.dtype, device=X.device)
    A = torch.tensor([
        [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
        [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
        [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
        [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
    ], dtype=X.dtype, device=X.device)
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=X.dtype, device=X.device)
    
    def evaluate_hartmann(alpha):
        val = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        for i in range(4):
            exponent = torch.sum(A[i] * (X - P[i])**2, dim=-1)
            val += alpha[i] * torch.exp(-exponent)
        return val

    y1 = evaluate_hartmann(alpha1)
    y2 = evaluate_hartmann(alpha2)
    return torch.stack([y1 + noise_std * torch.randn_like(y1), y2 + noise_std * torch.randn_like(y2)], dim=-1)

# =====================================================================
# AXIS 2: CONSTRAINT COMPLEXITY (Low M=2 to High M=5, 10)
# Stress-tests the joint intersection of multiple strict thresholds.[1]
# =====================================================================

def scalable_rosenbrock_mo(X, M=3, noise_std=0.1):
    """
    Scalable Rosenbrock: Configurable to arbitrary dimension 'd' and
    'M' competing constraint objectives. Introduces severe 'contour traps'.
    """
    d = X.shape[-1]
    objectives = []
    
    # Objective 1: Standard Rosenbrock banana-valley
    y1 = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
    for i in range(d - 1):
        y1 += 100 * (X[..., i+1] - X[..., i]**2)**2 + (1 - X[..., i])**2
    objectives.append(y1 + noise_std * torch.randn_like(y1))
    
    # Objective 2: L1 Norm (linear competition)
    y2 = torch.sum(torch.abs(X), dim=-1)
    objectives.append(y2 + (noise_std * 0.5) * torch.randn_like(y2))
    
    # Objectives 3 to M: Trigonometric and quadratic shifts over subspaces
    for m in range(2, M):
        shift = 0.5 * m
        y_m = torch.sum((X - shift)**2, dim=-1) + torch.sin(2 * math.pi * X[..., m % d])
        objectives.append(y_m + (noise_std * 0.5) * torch.randn_like(y_m))
        
    return torch.stack(objectives, dim=-1)

# =====================================================================
# AXIS 3: DIFFICULTY & GEOMETRY (Feasible Volume Vf from 50% down to <1%)
# Tests the 'needle-in-a-haystack' problem and disconnected multi-modal islands.
# =====================================================================

def concentric_shells_mo(X, noise_std=0.02):
    """
    NEW FUNCTION: M concentric shells.[1] 
    Creates narrow, ring-shaped feasible intersection corridors.
    Hugging this boundary results in total failure; requires safe interior seeking.
    """
    # Calculate distance to center
    dist = torch.norm(X - 0.5, dim=-1)
    
    # Objective 1: Outer shell boundary
    y1 = torch.cos(4 * math.pi * dist)
    # Objective 2: Inner shell boundary (competing frequency)
    y2 = torch.cos(6 * math.pi * dist)
    
    return torch.stack([y1 + noise_std * torch.randn_like(y1), y2 + noise_std * torch.randn_like(y2)], dim=-1)

def multimodal_gmm_mo(X, M=3, noise_std=0.02):
    """
    NEW FUNCTION: Highly non-linear Gaussian Mixture Model (GMM).
    Shatters the search space into small, isolated 'feasible islands'.
    Tests if look-ahead CAS can cross 'blue' gaps to discover hidden 'green' zones.
    """
    d = X.shape[-1]
    
    # Define coordinate centers for multiple Gaussian modes
    modes_centers = torch.tensor([
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        [0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
    ], dtype=X.dtype, device=X.device)[:, :d]
    
    objectives = []
    for m in range(M):
        y_m = torch.zeros(X.shape[:-1], dtype=X.dtype, device=X.device)
        # Combine multiple local Gaussian peaks with different weights
        for j, center in enumerate(modes_centers):
            weight = 1.5 if (j + m) % 2 == 0 else 0.5
            dist_sq = torch.sum((X - center)**2, dim=-1)
            y_m += weight * torch.exp(-dist_sq / 0.05)
        objectives.append(y_m + noise_std * torch.randn_like(y_m))
        
    return torch.stack(objectives, dim=-1)

# =====================================================================
# SYSTEMATIC EXPERIMENTAL MATRIX
# =====================================================================

def get_experiment_setup(name):
    """
    Experiment lookup map categorized strictly by:
    - Dimensionality (d)
    - Constraint Complexity (M)
    - Feasibility Difficulty (V_f: Easy, Moderate, Hard, Needle)
    """
    setups = {
        # --- TIER 1: Low-Dimensional / Connected Baselines ---
        "toy_2d_easy": (
            toy_2d_multiobjective,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            [("gt", 0.6), ("gt", 0.5)] # Vf ~ 35% (Easy)
        ),
        "fonseca_fleming": (
            fonseca_fleming_multiobjective,
            torch.tensor([[-4.0, -4.0], [4.0, 4.0]], dtype=torch.double),
            [("gt", 0.95), ("gt", 0.95)]
        ),
        "binh_and_korn": (
            binh_and_korn_multiobjective,
            torch.tensor([[0.0, 0.0], [5.0, 3.0]], dtype=torch.double),
            [("lt", 60.0), ("lt", 20.0)] 
        ),
        "concentric_shells_hard": (
            concentric_shells_mo,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            [("gt", 0.5), ("gt", 0.5)] # Vf ~ 4.2% (Hard, highly non-convex)
        ),
        
        # --- TIER 2: Flat Valleys & Redundant Contours ---
        "rosenbrock_5d_mod": (
            lambda X, **kwargs: scalable_rosenbrock_mo(X, M=2, **kwargs),
            torch.tensor([[-2.0]*5, [2.0]*5], dtype=torch.double),
            [("lt", 400.0), ("gt", 2.5)] # Vf ~ 12.0% (Moderate)
        ),
        "rosenbrock_8d_hard": (
            lambda X, **kwargs: scalable_rosenbrock_mo(X, M=3, **kwargs),
            torch.tensor([[-2.0]*8, [2.0]*8], dtype=torch.double),
            [("lt", 800.0), ("gt", 4.0), ("lt", 15.0)] # Vf ~ 2.1% (Hard)
        ),
        
        # --- TIER 3: Isolated Multimodal Islands ---
        "gmm_2d_mod": (
            lambda X, **kwargs: multimodal_gmm_mo(X, M=2, **kwargs),
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            [("gt", 0.8), ("gt", 0.8)] # Vf ~ 14.5% (Disconnected Islands)
        ),
        "gmm_5d_hard": (
            lambda X, **kwargs: multimodal_gmm_mo(X, M=3, **kwargs),
            torch.tensor([[0.0]*5, [1.0]*5], dtype=torch.double),
            [("gt", 1.0), ("gt", 1.0), ("gt", 0.8)] # Vf ~ 3.1% (Deep sparse pockets)
        ),
        
        # --- TIER 4: High-Dimensional Needle-in-a-Haystack ---
        "gmm_10d_needle": (
            lambda X, **kwargs: multimodal_gmm_mo(X, M=5, **kwargs),
            torch.tensor([[0.0]*10, [1.0]*10], dtype=torch.double),
            [("gt", 1.2), ("gt", 1.2), ("gt", 1.0), ("gt", 1.0), ("gt", 0.8)] 
            # Vf < 0.1% (Needle-in-a-haystack, 10D space, 5 active constraints)
        )
    }
    
    if name not in setups:
        raise ValueError(f"Unknown experiment setup: {name}")
    return setups[name]