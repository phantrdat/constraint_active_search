import torch
import math

def toy_2d_multiobjective(X):
    """
    A simple 2D synthetic problem with 2 competing outputs.
    X: Tensor of shape (..., 2)
    Returns: Tensor of shape (..., 2)
    """
    y1 = 1 - torch.norm(X - 0.25, dim=-1) ** 2
    y2 = 1 - torch.norm(X - 0.75, dim=-1) ** 2
    
    # Add a small amount of noise to simulate real-world conditions
    y1 += 0.05 * torch.randn_like(y1)
    y2 += 0.05 * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def yf2d_multiobjective(X):
    """A 2D function with two different Gaussian-like peaks."""
    v = torch.exp(-2 * (X[..., 0] - 0.3) ** 2 - 4 * (X[..., 1] - 0.6) ** 2)
    return torch.stack((v, v), dim=-1)

def six_hump_camel_multiobjective(X):
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
    y1 += 0.05 * torch.randn_like(y1)
    y2 += 0.05 * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)

def binh_and_korn_multiobjective(X):
    """
    Binh and Korn function.
    X: Tensor of shape (..., 2)
    """
    x1 = X[..., 0]
    x2 = X[..., 1]
    
    y1 = 4 * x1**2 + 4 * x2**2
    y2 = (x1 - 5)**2 + (x2 - 5)**2
    
    y1 += 0.5 * torch.randn_like(y1)
    y2 += 0.5 * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def fonseca_fleming_multiobjective(X):
    """
    Fonseca-Fleming function (2D).
    X: Tensor of shape (..., 2)
    """
    term1 = (X[..., 0] - 1/torch.sqrt(torch.tensor(2.0)))**2 + (X[..., 1] - 1/torch.sqrt(torch.tensor(2.0)))**2
    term2 = (X[..., 0] + 1/torch.sqrt(torch.tensor(2.0)))**2 + (X[..., 1] + 1/torch.sqrt(torch.tensor(2.0)))**2
    
    y1 = 1 - torch.exp(-term1)
    y2 = 1 - torch.exp(-term2)
    
    y1 += 0.01 * torch.randn_like(y1)
    y2 += 0.01 * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def kursawe_multiobjective(X):
    """
    Kursawe function (2D).
    X: Tensor of shape (..., 2)
    """
    x1 = X[..., 0]
    x2 = X[..., 1]
    
    y1 = -10 * torch.exp(-0.2 * torch.sqrt(x1**2 + x2**2))
    y2 = torch.abs(x1)**0.8 + 5 * torch.sin(x1**3) + torch.abs(x2)**0.8 + 5 * torch.sin(x2**3)
    
    y1 += 0.05 * torch.randn_like(y1)
    y2 += 0.05 * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def hartmann6_multiobjective(X):
    """
    Hartmann 6-D function with two competing objectives.
    X: Tensor of shape (..., 6)
    """
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=X.dtype, device=X.device)
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], dtype=X.dtype, device=X.device)
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=X.dtype, device=X.device)

    # Objective 1
    inner_sum = torch.sum(A * (X.unsqueeze(-2) - P)**2, dim=-1)
    y1 = torch.sum(alpha * torch.exp(-inner_sum), dim=-1)

    # Objective 2: Reverse alpha to create a competing landscape
    alpha2 = torch.tensor([3.2, 3.0, 1.2, 1.0], dtype=X.dtype, device=X.device)
    y2 = torch.sum(alpha2 * torch.exp(-inner_sum), dim=-1)

    y1 += 0.01 * torch.randn_like(y1)
    y2 += 0.01 * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)

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

    y1 += noise_std * torch.randn_like(y1)
    y2 += noise_std * torch.randn_like(y2)
    
    return torch.stack([y1, y2], dim=-1)

def rosenbrock_multiobjective(X):
    """
    Rosenbrock function (5D) with two objectives.
    X: Tensor of shape (..., 5)
    """
    x_i = X[..., :-1]
    x_next = X[..., 1:]
    y1 = torch.sum(100 * (x_next - x_i**2)**2 + (1 - x_i)**2, dim=-1)
    
    # y2: Sum of absolute values (linear competition)
    y2 = torch.sum(torch.abs(X), dim=-1)
    
    y1 += 0.1 * torch.randn_like(y1)
    y2 += 0.05 * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)

def styblinski_tang_multiobjective(X):
    """
    Styblinski-Tang function (5D) with two objectives.
    X: Tensor of shape (..., 5)
    """
    # Standard Styblinski-Tang
    y1 = 0.5 * torch.sum(X**4 - 16 * X**2 + 5 * X, dim=-1)
    
    # y2: Quadratic bowl centered away from the ST global minimum
    y2 = torch.sum((X + 2.5)**2, dim=-1)
    
    y1 += 0.5 * torch.randn_like(y1)
    y2 += 0.5 * torch.randn_like(y2)
    return torch.stack([y1, y2], dim=-1)

def griewank8_multiobjective(X):
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
    return res + 0.05 * torch.randn_like(res)

def levy4_multiobjective(X):
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
    return res + 0.02 * torch.randn_like(res)

def dixon_price6_multiobjective(X):
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
    return res + 0.1 * torch.randn_like(res)

def michalewicz10_multiobjective(X):
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
    return res + 0.01 * torch.randn_like(res)

def get_experiment_setup(name="toy_2d"):
    """Returns the function, domain bounds, and threshold constraints for a given experiment."""
    if name == "toy_2d":
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        constraints = [("gt", 0.6), ("gt", 0.5)]
        return toy_2d_multiobjective, bounds, constraints
    elif name == "yf2d":
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        constraints = [("lt", 0.75), ("gt", 0.55)]
        return yf2d_multiobjective, bounds, constraints
    elif name == "six_hump_camel":
        bounds = torch.tensor([[-2.0, -1.0], [2.0, 1.0]], dtype=torch.double)
        constraints = [("lt", -1.0), ("gt", 1.5)]
        return six_hump_camel_multiobjective, bounds, constraints
    elif name == "binh_and_korn":
        bounds = torch.tensor([[0.0, 0.0], [5.0, 3.0]], dtype=torch.double)
        constraints = [("lt", 80.0), ("lt", 40.0)]
        return binh_and_korn_multiobjective, bounds, constraints
    elif name == "fonseca_fleming":
        bounds = torch.tensor([[-4.0, -4.0], [4.0, 4.0]], dtype=torch.double)
        constraints = [("lt", 0.95), ("lt", 0.95)]
        return fonseca_fleming_multiobjective, bounds, constraints
    elif name == "kursawe":
        bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]], dtype=torch.double)
        constraints = [("lt", -4.0), ("lt", 5.0)]
        return kursawe_multiobjective, bounds, constraints
    elif name == "hartmann6":
        bounds = torch.tensor([[0.0] * 6, [1.0] * 6], dtype=torch.double)
        constraints = [("gt", 1.5), ("gt", 1.5)]
        return hartmann6_multiobjective, bounds, constraints
    elif name == "ackley10":
        bounds = torch.tensor([[-5.0] * 10, [5.0] * 10], dtype=torch.double)
        constraints = [("lt", 10.0), ("lt", 10.0)]
        return ackley_multiobjective, bounds, constraints
    elif name == "rosenbrock5":
        bounds = torch.tensor([[-2.0] * 5, [2.0] * 5], dtype=torch.double)
        constraints = [("lt", 500.0), ("gt", 2.0)]
        return rosenbrock_multiobjective, bounds, constraints
    elif name == "styblinski_tang5":
        bounds = torch.tensor([[-5.0] * 5, [5.0] * 5], dtype=torch.double)
        constraints = [("lt", -50.0), ("lt", 100.0)]
        return styblinski_tang_multiobjective, bounds, constraints
    elif name == "griewank8":
        bounds = torch.tensor([[-5.0] * 8, [5.0] * 8], dtype=torch.double)
        constraints = [("lt", 2.0), ("lt", 2.0), ("lt", 10.0), ("lt", 15.0)]
        return griewank8_multiobjective, bounds, constraints
    elif name == "levy4":
        bounds = torch.tensor([[-5.0] * 4, [5.0] * 4], dtype=torch.double)
        constraints = [("lt", 5.0), ("lt", 5.0), ("gt", 0.0)]
        return levy4_multiobjective, bounds, constraints
    elif name == "dixon_price6":
        bounds = torch.tensor([[-10.0] * 6, [10.0] * 6], dtype=torch.double)
        constraints = [("lt", 500.0), ("lt", 50.0), ("gt", -2.0)]
        return dixon_price6_multiobjective, bounds, constraints
    elif name == "michalewicz10":
        bounds = torch.tensor([[0.0] * 10, [math.pi] * 10], dtype=torch.double)
        constraints = [("lt", -5.0), ("lt", -3.0), ("lt", 50.0), ("lt", 20.0), ("gt", -5.0)]
        return michalewicz10_multiobjective, bounds, constraints
    raise ValueError(f"Unknown experiment setup: {name}")
