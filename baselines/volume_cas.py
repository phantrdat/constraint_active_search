import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

class VolumeCAS(AcquisitionFunction):
    def __init__(self, model, constraints, beta=1.96**2, gamma=2.0):
        super().__init__(model=model)
        self.constraints = constraints
        self.beta = beta
        self.gamma = gamma  # Strictness parameter for feasibility

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        
        transformed_mean = torch.empty_like(mean)
        # Standardize constraints so that "feasible" always translates to >= 0
        for i, (direction, threshold) in enumerate(self.constraints):
            if direction == "gt":
                transformed_mean[..., i] = mean[..., i] - threshold
            elif direction == "lt":
                transformed_mean[..., i] = threshold - mean[..., i]
                
        beta_sqrt = torch.sqrt(torch.tensor(self.beta, dtype=X.dtype, device=X.device))
        
        u = transformed_mean + beta_sqrt * std
        l = transformed_mean - beta_sqrt * std
        
        # 1. Compute Total Volume U(x)
        vol_B = torch.prod(u - l, dim=-1)
        
        # 2. Compute Volume of the Positive Orthant Partition B_1(x)
        u_trunc = torch.clamp(u, min=0.0)
        l_trunc = torch.clamp(l, min=0.0)
        vol_B1 = torch.prod(u_trunc - l_trunc, dim=-1)
        
        # 3. Asymmetric Volume Score for CAS
        # Calculate what fraction of the confidence box lies in the feasible region
        eps = 1e-9 # Prevent division by zero
        feasibility_ratio = vol_B1 / (vol_B + eps)
        
        # Scale the total uncertainty by the feasibility ratio raised to gamma
        a = vol_B * (feasibility_ratio ** self.gamma)
        
        return a.squeeze(-1)