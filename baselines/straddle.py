import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

class Straddle(AcquisitionFunction):
    def __init__(self, model, constraints, beta=1.96**2, lambda_penalty=1.0):
        super().__init__(model=model)
        self.constraints = constraints
        self.beta = beta
        self.lambda_penalty = lambda_penalty

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
        
        vol_B = torch.prod(u - l, dim=-1)
        
        u_trunc = torch.clamp(u, min=0.0)
        l_trunc = torch.clamp(l, min=0.0)
        vol_B1 = torch.prod(u_trunc - l_trunc, dim=-1)
        
        D = torch.abs(2 * vol_B1 - vol_B)
        a = vol_B - self.lambda_penalty * D
        
        return a.squeeze(-1)