import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

class LookaheadCAS(AcquisitionFunction):
    """
    Implements the 1-step look-ahead Constraint Active Search, combining
    the VolumeCAS feasibility ratio with a novelty-promoting term.
    The acquisition function is: α(x) = (Vol(B1(x)) / Vol(B(x)))^γ * Novel_Volume(x)
    """
    def __init__(self, model, constraints, train_inputs, beta=3.84, gamma=2.0, lambda_novelty=1.0):
        """
        Args:
            model: A fitted BoTorch model.
            constraints: A list of (direction, threshold) tuples.
            train_inputs: Tensor of previously evaluated points in the input space.
            beta: Controls the size of the confidence region. The confidence
                interval is mu +/- sqrt(beta) * std. Default is 1.96**2.
            gamma: Exponent for the feasibility ratio.
            lambda_novelty: Weight for the novelty-promoting term.
        """
        super().__init__(model=model)
        self.constraints = constraints
        self.train_inputs = train_inputs
        self.beta = beta
        self.gamma = gamma
        self.lambda_novelty = lambda_novelty
        
        self.register_buffer("_thresholds", torch.tensor([thresh for _, thresh in constraints]))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate the LookaheadCAS acquisition function on candidate set X."""
        self._thresholds = self._thresholds.to(X)
        posterior = self.model.posterior(X)
        mus = posterior.mean
        std = posterior.variance.sqrt()
        beta_val = self.beta**0.5

        # --- VolumeCAS Score (Feasibility Ratio) ---
        lcb = mus - beta_val * std
        ucb = mus + beta_val * std

        vol_b = (ucb - lcb).prod(dim=-1)

        feasible_lcb = lcb.clone()
        feasible_ucb = ucb.clone()

        for i, (direction, threshold) in enumerate(self.constraints):
            if direction == "gt":
                feasible_lcb[..., i] = torch.max(lcb[..., i], self._thresholds[i])
            else: # "lt"
                feasible_ucb[..., i] = torch.min(ucb[..., i], self._thresholds[i])
        
        vol_b1 = (feasible_ucb - feasible_lcb).clamp(min=0.0).prod(dim=-1)

        ratio = vol_b1 / (vol_b + 1e-9)
        volume_cas_score = ratio.pow(self.gamma)

        # --- Novelty Score ---
        if self.train_inputs is not None and self.train_inputs.shape[0] > 0:
            min_dist = torch.cdist(X.squeeze(1), self.train_inputs).min(dim=-1).values
        else:
            min_dist = torch.full((X.shape[0],), 1e6, dtype=X.dtype, device=X.device)

        novelty_score = self.lambda_novelty * min_dist

        # --- Final Acquisition Value ---
        acq_value = volume_cas_score * novelty_score.unsqueeze(-1)
        
        return acq_value.squeeze(-1)