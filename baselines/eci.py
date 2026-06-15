import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.transforms import t_batch_mode_transform

def smooth_mask(x, a, eps=2e-3):
    """Returns 0ish for x < a and 1ish for x > a."""
    return torch.nn.Sigmoid()((x - a) / eps)

def smooth_box_mask(x, a, b, eps=2e-3):
    """Returns 1ish for a < x < b and 0ish otherwise."""
    return smooth_mask(x, a, eps) - smooth_mask(x, b, eps)

class ExpectedCoverageImprovement(MCAcquisitionFunction):
    def __init__(self, model, constraints, punchout_radius, bounds, train_inputs, num_samples=128, **kwargs):
        super().__init__(model=model, objective=IdentityMCObjective(), **kwargs)
        self.constraints = constraints
        self.punchout_radius = punchout_radius
        self.bounds = bounds
        self.base_points = train_inputs
        self.dim = bounds.shape[1]
        
        # Generate a ball of points to be used for Monte Carlo integration
        self.ball_of_points = self._generate_ball_of_points(
            num_samples=num_samples, radius=punchout_radius, 
            device=bounds.device, dtype=bounds.dtype
        )
        self._thresholds = torch.tensor([threshold for _, threshold in self.constraints]).to(bounds)

    def _generate_ball_of_points(self, num_samples, radius, device=None, dtype=torch.double):
        tkwargs = {"device": device, "dtype": dtype}
        z = torch.randn(num_samples, self.dim, **tkwargs)
        z = z / torch.norm(z, dim=-1, keepdim=True)
        r = torch.rand(num_samples, 1, **tkwargs) ** (1 / self.dim)
        return radius * r * z

    def _get_base_point_mask(self, X):
        """Calculates distance to previous observations to avoid redundant sampling."""
        distance_matrix = torch.cdist(X, self.base_points)
        return smooth_mask(distance_matrix, self.punchout_radius)

    def _estimate_probabilities_of_satisfaction_at_points(self, points):
        """Estimate the probability of satisfying the given constraints using Normal CDF."""
        posterior = self.model.posterior(X=points)
        mus, sigma2s = posterior.mean, posterior.variance
        dist = torch.distributions.normal.Normal(mus, sigma2s.sqrt())
        norm_cdf = dist.cdf(self._thresholds)
        
        probs = torch.ones(points.shape[:-1]).to(points)
        for i, (direction, _) in enumerate(self.constraints):
            probs = probs * (norm_cdf[..., i] if direction == "lt" else 1 - norm_cdf[..., i])
        return probs

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """Evaluate ECI on candidate set X."""
        ball_around_X = self.ball_of_points + X
        
        domain_mask = smooth_box_mask(ball_around_X, self.bounds[0, :], self.bounds[1, :]).prod(dim=-1)
        num_points_in_integral = domain_mask.sum(dim=-1)
        base_point_mask = self._get_base_point_mask(ball_around_X).prod(dim=-1)
        
        prob = self._estimate_probabilities_of_satisfaction_at_points(ball_around_X)
        masked_prob = prob * domain_mask * base_point_mask
        
        y = masked_prob.sum(dim=-1) / torch.clamp(num_points_in_integral, min=1.0)
        return y