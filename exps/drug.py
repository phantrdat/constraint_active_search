import torch
import pandas as pd
import numpy as np
import os
import glob

class DrugExperiment:
    """
    Updated DrugExperiment to load and merge multiple data parts into a single pool.
    """
    def __init__(self, target_name="3CLPro", data_dir="data/DrugImprover/data", num_data_parts=20, device=None):
        self.target_name = target_name
        self.data_dir = data_dir
        self.num_data_parts = num_data_parts
        self.device = device if device is not None else torch.device("cpu")

        print(f"Initializing DrugExperiment for {target_name} using {num_data_parts} parts...")
        self._load_data()
        self._normalize_and_set_constraints()
        print("DrugExperiment initialized successfully.")

    def _load_data(self):
        """
        Loads and joins all preprocessed molecule data parts from 1 to num_data_parts.
        """
        cache_dir = os.path.join(self.data_dir, "cache")
        self.dataframe = pd.read_pickle(os.path.join(cache_dir, f"{self.target_name}_selected_features.pkl"))

        # Concatenate all loaded dataframes
        print(f"Total molecules loaded across all parts: {len(self.dataframe)}")

        # Extract X_pool (descriptors) and Y_pool (objectives)
        desc_cols = [col for col in self.dataframe.columns if col.startswith('descriptor_')]
        obj_cols = [col for col in self.dataframe.columns if col.startswith('objective_')]
        
        self.X_pool = torch.tensor(self.dataframe[desc_cols].values, device=self.device, dtype=torch.double)
        
        self.Y_pool = torch.tensor(self.dataframe[obj_cols].values, device=self.device, dtype=torch.double)
        self.bounds = torch.stack([self.X_pool.min(0).values, self.X_pool.max(0).values]).to(self.device)

    def _normalize_and_set_constraints(self):
        """Normalizes inputs and objectives via Min-Max and sets feasibility thresholds."""
        if self.Y_pool.shape[0] == 0:
            raise ValueError("No data available.")

        # 1. Normalize Inputs (X_pool) to be in [0, 1]
        # x_min = self.X_pool.min(dim=0).values
        # x_max = self.X_pool.max(dim=0).values
        # # Add epsilon to avoid division by zero
        # self.X_pool = (self.X_pool - x_min) / (x_max - x_min + 1e-6)
        # self.bounds = torch.tensor([[0.0] * self.X_pool.shape[1], [1.0] * self.X_pool.shape[1]], dtype=torch.double)

        # 2. Normalize Objectives (Y_pool)
        self.Y_pool_normalized = self.Y_pool.clone()

        # Indices for objectives to be normalized: docking, esol, sa_score
        # These correspond to objective_1, objective_2, and objective_3
        indices_to_normalize = [0, 1, 2] 

        # Perform Min-Max Normalization on specified columns
        y_to_norm = self.Y_pool[:, indices_to_normalize]
        y_min = y_to_norm.min(dim=0).values
        y_max = y_to_norm.max(dim=0).values
        
        # Add epsilon to avoid division by zero if min == max
        self.Y_pool_normalized[:, indices_to_normalize] = (y_to_norm - y_min) / (y_max - y_min + 1e-6)

        # 3. Set thresholds to define a feasible set of a specific size (~30%).
        # To do this, we find the minimum normalized score for each molecule across all objectives.
        # This score represents the "worst-case" performance for that molecule.
        min_scores_per_molecule = self.Y_pool_normalized.min(dim=1).values

        # We then find the threshold value that corresponds to the 70th percentile of these "worst-case" scores.
        # This ensures that molecules in the top 30% (those with a min_score >= threshold) form our feasible set.
        feasible_quantile = 0.7
        joint_threshold = torch.quantile(min_scores_per_molecule, feasible_quantile)
        self.thresholds = torch.full_like(self.Y_pool_normalized[0], joint_threshold)
        self.constraints = [("gt", t.item()) for t in self.thresholds]
        
        # 4. Identify feasible molecules (those meeting ALL 5 criteria)
        is_feasible = (self.Y_pool_normalized >= self.thresholds).all(dim=1)
        
        self.S_pool_X = self.X_pool[is_feasible]
        self.S_pool_Y = self.Y_pool[is_feasible] # Original scale
        self.S_pool_Y_normalized = self.Y_pool_normalized[is_feasible] # Normalized scale
        
        print(f"Normalization complete for objectives 1, 2, and 3.")
        print(f"Found {len(self.S_pool_X)} feasible molecules ({len(self.S_pool_X)/len(self.X_pool):.2%})")

    def __call__(self, X):
        """Finds nearest neighbors in the combined pool."""
        dist_matrix = torch.cdist(X, self.X_pool)
        nn_indices = torch.argmin(dist_matrix, dim=1)
        return self.Y_pool_normalized[nn_indices]

# Adjusted factory function (removed data_part_idx as it's no longer needed for loading)
def get_experiment_setup(name="3CLPro", data_dir="data/DrugImprover/data", num_data_parts=1, device=None, **kwargs):
    if name in ["3CLPro", "6T2W", "RTCB", "WRN"]:
        task = DrugExperiment(target_name=name, data_dir=data_dir, num_data_parts=num_data_parts, device=device, **kwargs)
        return task, task.bounds, task.constraints
    
    # Fallback to synthetic functions if name doesn't match
    try:
        from exps.synthetic import get_experiment_setup as get_synthetic_setup
        print(f"'{name}' not a drug target, falling back to synthetic experiments.")
        return get_synthetic_setup(name)
    except ImportError:
        raise ValueError(f"Unknown experiment setup: {name}")
