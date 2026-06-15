import torch
from botorch.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

class SingleObjectiveStraddle(AcquisitionFunction):
    def __init__(self, model, threshold, beta=4.0):
        """
        Single-objective Straddle acquisition function for active search.
        
        Calculates the score as: beta * sigma(x) - |mu(x) - threshold|.
        This balances exploring regions of high uncertainty and regions near 
        the target threshold.

        Args:
            model: A fitted single-task GP model.
            threshold: The target threshold for the objective.
            beta: Confidence parameter scaling the uncertainty (std dev).
        """
        super().__init__(model=model)
        self.threshold = threshold
        self.beta = beta

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        posterior = self.model.posterior(X)
        # Squeeze the q-batch and output dimensions to return a [batch_shape] tensor
        mean = posterior.mean.view(X.shape[:-2]) 
        std = posterior.variance.sqrt().view(X.shape[:-2])
        
        return self.beta * std - torch.abs(mean - self.threshold)