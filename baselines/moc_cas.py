import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch.distributions import Normal

class MOCCAS(AcquisitionFunction):
    def __init__(self, model, constraints, previous_y, radius=0.1, beta=2.0, eps=1e-3, 
                 lambda_repulse=0.2, current_iteration=0, beta_anneal_t=150, beta_anneal_factor=0.7):
        """
        Multi-Objective Coverage via Constraint Active Search (MOC-CAS).

        Args:
            model: A fitted BoTorch model.
            constraints: A list of (direction, threshold) tuples.
            previous_y: Tensor of previously evaluated objective values.
            radius: The coverage radius in the objective space.
            beta: The base UCB confidence parameter for annealing.
            eps: Smoothing parameter for the probit feasibility gate.
            lambda_repulse: Weight for the diversity-promoting repulsion term.
            current_iteration: The current BO iteration, for beta annealing.
            beta_anneal_t: Iteration to start annealing beta.
            beta_anneal_factor: Multiplicative factor for beta annealing.
        """
        super().__init__(model=model)
        self.constraints = constraints
        self.previous_y = previous_y
        self.radius = radius
        self.beta = beta
        self.eps = eps
        self.lambda_repulse = lambda_repulse
        self.current_iteration = current_iteration
        self.beta_anneal_t = beta_anneal_t
        self.beta_anneal_factor = beta_anneal_factor
        
        # Store thresholds as a buffer for automatic device placement
        self.register_buffer("_thresholds", torch.tensor([thresh for _, thresh in constraints]))

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate the MOC-CAS acquisition function on candidate set X."""
        # 1. Optimistic Objective Estimation (UCB) with beta annealing.
        beta_t = self.beta * torch.log1p(torch.tensor(self.current_iteration + 1, dtype=X.dtype, device=X.device))
        if self.current_iteration > self.beta_anneal_t:
            beta_t *= self.beta_anneal_factor

        posterior = self.model.posterior(X)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        ucb = mean + beta_t * std  # Shape: (batch_shape, 1, num_objectives)
        
        self._thresholds = self._thresholds.to(X)
        
        # 2. Smooth Feasibility Gate: Calculate probability of satisfying constraints.
        normal = Normal(0, 1)
        feasibility_score = torch.ones(X.shape[:-1], dtype=X.dtype, device=X.device)
        
        for i, (direction, _) in enumerate(self.constraints):
            z = (ucb[..., i] - self._thresholds[i]) / self.eps
            if direction == "lt":
                z = -z
            prob_feasible = normal.cdf(z)
            feasibility_score = feasibility_score * prob_feasible
        
        # Identify previously observed feasible points for coverage and diversity calculations.
        previous_feasible_y_mask = torch.ones(self.previous_y.shape[0], dtype=torch.bool, device=X.device)
        for i, (direction, threshold) in enumerate(self.constraints):
            if direction == "lt":
                previous_feasible_y_mask &= (self.previous_y[:, i] < threshold)
            else:
                previous_feasible_y_mask &= (self.previous_y[:, i] > threshold)
        previous_feasible_y = self.previous_y[previous_feasible_y_mask]

        # 3. Coverage Gain: Reward points in low-density regions of the feasible objective space.
        if previous_feasible_y is not None and len(previous_feasible_y) > 0:
            # Calculate soft kernel density based on distance to previous feasible points.
            dist_sq = torch.cdist(ucb, previous_feasible_y.to(X)) ** 2
            covered_density = torch.exp(-dist_sq / (4 * self.radius ** 2)).sum(dim=-1)
            
            # Gain is inversely related to density (high gain for novel points).
            gain_from_coverage = torch.exp(-covered_density)
        else:
            gain_from_coverage = torch.ones_like(feasibility_score)

        # 4. Diversity Term: Promote points far from existing feasible observations.
        repulsion_score = torch.zeros_like(feasibility_score)
        if previous_feasible_y is not None and len(previous_feasible_y) > 0:
            # Calculate the minimum distance to any previous feasible point.
            min_dist_to_feasible = torch.cdist(ucb, previous_feasible_y.to(X)).min(dim=-1).values
            # Scale and clamp the repulsion score for stability.
            repulsion_score = min_dist_to_feasible / self.radius 
            repulsion_score = torch.clamp(repulsion_score, max=5.0)
        else:
            # No feasible points yet, so no repulsion.
            repulsion_score = torch.ones_like(feasibility_score)

        # 5. Final Score: Combine feasibility, coverage gain, and diversity.
        # The score is the feasibility probability multiplied by the sum of coverage gain and a weighted diversity term.
        acq_value = feasibility_score * (gain_from_coverage + self.lambda_repulse * repulsion_score)
        
        return acq_value.squeeze(-1)
    