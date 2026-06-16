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

def yf2d_multiobjective(X, noise_std=0.0):
    """A 2D function with two different Gaussian-like peaks."""
    v = torch.exp(-2 * (X[..., 0] - 0.3) ** 2 - 4 * (X[..., 1] - 0.6) ** 2)
    res = torch.stack((v, v), dim=-1)
    return res + noise_std * torch.randn_like(res)

def six_hump_camel_multiobjective(X, noise_std=0.05):
    """
    The Six-hump camel function, modified to have two outputs.
    It has two global minima at (0.0898, -0.7126) and (-0.0898, 0.7126).
    """
    x1 = X[..., 0]
    x2 = X[..., 1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    y1 = term1 + term2 + term3
    y2 = torch.sin(x1 * 3) + torch.cos(x2 * 3)
    res = torch.stack([y1, y2], dim=-1)
    return res + noise_std * torch.randn_like(res)

def kursawe_multiobjective(X, noise_std=0.05):
    """
    Kursawe function (2D).
    X: Tensor of shape (..., 2)
    """
    x1 = X[..., 0]
    x2 = X[..., 1]
    
    y1 = -10 * torch.exp(-0.2 * torch.sqrt(x1**2 + x2**2))
    y2 = torch.abs(x1)**0.8 + 5 * torch.sin(x1**3) + torch.abs(x2)**0.8 + 5 * torch.sin(x2**3)
    
    res = torch.stack([y1, y2], dim=-1)
    return res + noise_std * torch.randn_like(res)

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

def ackley_multiobjective(X, noise_std=0.05):
    """
    Ackley function scaled to 10D with two objectives.
    X: Tensor of shape (..., 10)
    """
    d = X.shape[-1]
    a, b, c = 20, 0.2, 2 * math.pi
    
    sum1 = torch.sum(X**2, dim=-1)
    sum2 = torch.sum(torch.cos(c * X), dim=-1)
    y1 = -a * torch.exp(-b * torch.sqrt(sum1 / d)) - torch.exp(sum2 / d) + a + math.e
    
    # y2: Ackley centered at a different point (0.5 vector)
    X_shifted = X - 0.5
    sum1_s = torch.sum(X_shifted**2, dim=-1)
    sum2_s = torch.sum(torch.cos(c * X_shifted), dim=-1)
    y2 = -a * torch.exp(-b * torch.sqrt(sum1_s / d)) - torch.exp(sum2_s / d) + a + math.e

    res = torch.stack([y1, y2], dim=-1)
    return res + noise_std * torch.randn_like(res)

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

def rosenbrock_multiobjective(X, noise_std=0.1):
    """
    Rosenbrock function (5D) with two objectives.
    X: Tensor of shape (..., 5)
    """
    x_i = X[..., :-1]
    x_next = X[..., 1:]
    y1 = torch.sum(100 * (x_next - x_i**2)**2 + (1 - x_i)**2, dim=-1)
    
    # y2: Sum of absolute values (linear competition)
    y2 = torch.sum(torch.abs(X), dim=-1)
    
    res = torch.stack([y1, y2], dim=-1)
    return res + noise_std * torch.randn_like(res)

def styblinski_tang_multiobjective(X, noise_std=0.5):
    """
    Styblinski-Tang function (5D) with two objectives.
    X: Tensor of shape (..., 5)
    """
    # Standard Styblinski-Tang
    y1 = 0.5 * torch.sum(X**4 - 16 * X**2 + 5 * X, dim=-1)
    
    # y2: Quadratic bowl centered away from the ST global minimum
    y2 = torch.sum((X + 2.5)**2, dim=-1)
    
    res = torch.stack([y1, y2], dim=-1)
    return res + noise_std * torch.randn_like(res)

def griewank8_multiobjective(X, noise_std=0.05):
    """
    Griewank function (8D) with 4 objectives.
    X: Tensor of shape (..., 8)
    """
    d = X.shape[-1]
    
    # Obj 1: Standard Griewank
    sum_sq = torch.sum(X**2, dim=-1)
    prod_cos = torch.prod(torch.cos(X / torch.sqrt(torch.arange(1, d + 1, device=X.device, dtype=X.dtype))), dim=-1)
    y1 = 1 + sum_sq / 4000 - prod_cos
    
    # Obj 2: Shifted Griewank
    X_shifted = X - 1.0
    sum_sq_s = torch.sum(X_shifted**2, dim=-1)
    prod_cos_s = torch.prod(torch.cos(X_shifted / torch.sqrt(torch.arange(1, d + 1, device=X.device, dtype=X.dtype))), dim=-1)
    y2 = 1 + sum_sq_s / 4000 - prod_cos_s
    
    # Obj 3: Sphere
    y3 = torch.sum(X**2, dim=-1)
    
    # Obj 4: Linear competition (L1 norm)
    y4 = torch.sum(torch.abs(X), dim=-1)
    
    res = torch.stack([y1, y2, y3, y4], dim=-1)
    return res + noise_std * torch.randn_like(res)

def levy4_multiobjective(X, noise_std=0.02):
    """
    Levy function (4D) with 3 objectives.
    X: Tensor of shape (..., 4)
    """
    w = 1 + (X - 1.0) / 4.0
    term1 = torch.sin(math.pi * w[..., 0])**2
    term2 = torch.sum((w[..., :-1] - 1)**2 * (1 + 10 * torch.sin(math.pi * w[..., :-1] + 1)**2), dim=-1)
    term3 = (w[..., -1] - 1)**2 * (1 + torch.sin(2 * math.pi * w[..., -1])**2)
    y1 = term1 + term2 + term3
    
    # Obj 2: Shifted Levy
    w_s = 1 + (X - 0.5) / 4.0
    y2 = torch.sin(math.pi * w_s[..., 0])**2 + \
         torch.sum((w_s[..., :-1] - 1)**2 * (1 + 10 * torch.sin(math.pi * w_s[..., :-1] + 1)**2), dim=-1) + \
         (w_s[..., -1] - 1)**2 * (1 + torch.sin(2 * math.pi * w_s[..., -1])**2)
    
    # Obj 3: Sum of sines
    y3 = torch.sum(torch.sin(X), dim=-1)
    
    res = torch.stack([y1, y2, y3], dim=-1)
    return res + noise_std * torch.randn_like(res)

def dixon_price6_multiobjective(X, noise_std=0.1):
    """
    Dixon-Price function (6D) with 3 objectives.
    X: Tensor of shape (..., 6)
    """
    term1 = (X[..., 0] - 1)**2
    ii = torch.arange(2, 7, device=X.device, dtype=X.dtype)
    term2 = torch.sum(ii * (2 * X[..., 1:]**2 - X[..., :-1])**2, dim=-1)
    y1 = term1 + term2
    y2 = torch.sum((X - 0.5)**2, dim=-1)
    y3 = torch.sum(torch.cos(2 * math.pi * X), dim=-1)
    res = torch.stack([y1, y2, y3], dim=-1)
    return res + noise_std * torch.randn_like(res)

def michalewicz10_multiobjective(X, noise_std=0.01):
    """
    Michalewicz function (10D) with 5 objectives.
    X: Tensor of shape (..., 10)
    """
    m = 10
    ii = torch.arange(1, 11, device=X.device, dtype=X.dtype)
    def mich(inp):
        return -torch.sum(torch.sin(inp) * torch.sin(ii * inp**2 / math.pi)**(2 * m), dim=-1)
    
    y1, y2 = mich(X), mich(X - 1.0)
    y3, y4, y5 = torch.sum(X**2, dim=-1), torch.sum(torch.abs(X), dim=-1), torch.sum(torch.cos(X), dim=-1)
    res = torch.stack([y1, y2, y3, y4, y5], dim=-1)
    return res + noise_std * torch.randn_like(res)

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
        "toy_2d": (
            toy_2d_multiobjective,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            [("gt", 0.6), ("gt", 0.5)]
        ),
        "yf2d": (
            yf2d_multiobjective,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            [("lt", 0.75), ("gt", 0.55)]
        ),
        "six_hump_camel": (
            six_hump_camel_multiobjective,
            torch.tensor([[-2.0, -1.0], [2.0, 1.0]], dtype=torch.double),
            [("lt", -1.0), ("gt", 1.5)]
        ),
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
        "kursawe": (
            kursawe_multiobjective,
            torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.double),
            [("lt", -4.0), ("lt", 5.0)]
        ),
        "hartmann6": (
            hartmann6_multiobjective,
            torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double),
            [("gt", 1.5), ("gt", 1.5)]
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
        "rosenbrock5": (
            rosenbrock_multiobjective,
            torch.tensor([[-2.0] * 5, [2.0] * 5], dtype=torch.double),
            [("lt", 500.0), ("gt", 2.0)]
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
        "styblinski_tang5": (
            styblinski_tang_multiobjective,
            torch.tensor([[-5.0] * 5, [5.0] * 5], dtype=torch.double),
            [("lt", -50.0), ("lt", 100.0)]
        ),
        "griewank8": (
            griewank8_multiobjective,
            torch.tensor([[-5.0] * 8, [5.0] * 8], dtype=torch.double),
            [("lt", 15.0), ("lt", 25.0), ("lt", 30.0), ("lt", 15.0)]
        ),
        "levy4": (
            levy4_multiobjective,
            torch.tensor([[-5.0] * 4, [5.0] * 4], dtype=torch.double),
            [("lt", 5.0), ("lt", 5.0), ("gt", 0.0)]
        ),
        "ackley10": (
            ackley_multiobjective,
            torch.tensor([[-5.0] * 10, [5.0] * 10], dtype=torch.double),
            [("lt", 10.0), ("lt", 10.0)]
        ),
        
        # --- TIER 4: High-Dimensional Needle-in-a-Haystack ---
        "gmm_10d_needle": (
            lambda X, **kwargs: multimodal_gmm_mo(X, M=5, **kwargs),
            torch.tensor([[0.0]*10, [1.0]*10], dtype=torch.double),
            [("gt", 1.2), ("gt", 1.2), ("gt", 1.0), ("gt", 1.0), ("gt", 0.8)] 
            # Vf < 0.1% (Needle-in-a-haystack, 10D space, 5 active constraints)
        ),
        "michalewicz10": (
            michalewicz10_multiobjective,
            torch.tensor([[0.0] * 10, [math.pi] * 10], dtype=torch.double),
            [("lt", -5.0), ("lt", -3.0), ("lt", 50.0), ("lt", 20.0), ("gt", -5.0)]
        ),
    }
    
    if name not in setups:
        raise ValueError(f"Unknown experiment setup: {name}")
    return setups[name]