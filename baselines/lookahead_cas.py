import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


class LookAheadCAS(AcquisitionFunction):
    def __init__(
        self,
        model,
        constraints,
        train_X,
        grid_Z,
        candidate_set,
        radius=0.1,
        k=3,
        beta=3.84,
        gamma=2.0,
    ):
        super().__init__(model)
        self.constraints = constraints
        self.radius = radius
        self.k = k
        self.beta = beta
        self.gamma = gamma
        self.register_buffer("train_X", train_X)
        self.register_buffer("grid_Z", grid_Z)
        self.register_buffer("candidate_set", candidate_set)
        thresholds = torch.tensor(
            [t for _, t in constraints],
            dtype=train_X.dtype,
            device=train_X.device
        )
        self.register_buffer("thresholds", thresholds)

    ##################################################
    # posterior feasibility p(z)
    ##################################################
    def posterior_feasibility(self):
        posterior = self.model.posterior(self.grid_Z)
        mu = posterior.mean
        sigma = posterior.variance.sqrt()
        mu = mu.squeeze(-2)
        sigma = sigma.squeeze(-2)
        normal = torch.distributions.Normal(0, 1)
        probs = []
        for i, (direction, threshold) in enumerate(self.constraints):
            z = (mu[..., i] - threshold) / (sigma[..., i] + 1e-8)
            p = normal.cdf(z)
            if direction == "lt":
                p = 1 - p
            probs.append(p)
        probs = torch.stack(probs, dim=-1)
        return probs.prod(dim=-1)

    ##################################################
    # coverage mask
    ##################################################
    def coverage_mask(self, X):
        dist = torch.cdist(self.grid_Z, X)
        covered = (dist <= self.radius).any(dim=-1)
        return covered

    ##################################################
    # expected coverage U(A)
    ##################################################
    def expected_coverage(self, X, p_feas=None):
        if p_feas is None:
            p_feas = self.posterior_feasibility()
        covered = self.coverage_mask(X)
        return (p_feas * covered.float()).sum()

    ##################################################
    # Greedy completion
    ##################################################
    def greedy_complete(self, current_points, p_feas):
        A = current_points.clone()
        covered = self.coverage_mask(A)
        for _ in range(self.k - 1):
            dist = torch.cdist(self.grid_Z, self.candidate_set)
            candidate_cover = (dist <= self.radius)
            marginal = (candidate_cover & (~covered[:, None]))
            gains = (p_feas[:, None] * marginal.float()).sum(dim=0)
            idx = gains.argmax()
            xnew = self.candidate_set[idx].unsqueeze(0)
            A = torch.cat([A, xnew], dim=0)
            covered = covered | candidate_cover[:, idx]
        return A

    ##################################################
    # Forward
    ##################################################
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        Xcand = X.squeeze(1)
        p_feas = self.posterior_feasibility()
        U_before = self.expected_coverage(self.train_X, p_feas)
        acq_values = []
        for x in Xcand:
            x = x.unsqueeze(0)
            A0 = torch.cat([self.train_X, x], dim=0)
            A = self.greedy_complete(A0, p_feas)
            U_after = self.expected_coverage(A, p_feas)
            #
            # VolumeCAS feasibility term
            #
            posterior = self.model.posterior(x.unsqueeze(0))
            mu = posterior.mean.squeeze()
            sigma = posterior.variance.sqrt().squeeze()
            beta_sqrt = self.beta**0.5
            lcb = mu - beta_sqrt * sigma
            ucb = mu + beta_sqrt * sigma
            V = (ucb - lcb).prod()
            feasible_lcb = lcb.clone()
            feasible_ucb = ucb.clone()
            for i, (direction, threshold) in enumerate(self.constraints):
                if direction == "gt":
                    feasible_lcb[i] = max(lcb[i], threshold)
                else:
                    feasible_ucb[i] = min(ucb[i], threshold)
            Vplus = (feasible_ucb - feasible_lcb).clamp(min=0).prod()
            R = Vplus / (V + 1e-8)
            score = R.pow(self.gamma) * (U_after - U_before)
            acq_values.append(score)
        return torch.stack(acq_values)