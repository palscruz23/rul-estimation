from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class ResidualSimilarityModel:
    k: int = 50
    polys: List[np.ndarray] = None     # per-training-engine poly coeffs (degree 2)
    lengths: List[int] = None          # per-training-engine lengths

    def fit(self, train_fused: List[pd.DataFrame]) -> "ResidualSimilarityModel":
        self.polys = []
        self.lengths = []
        for df in train_fused:
            y = df["health_indicator"].values.astype(float)
            t = np.arange(len(y), dtype=float)
            # degree-2 polynomial fit
            p = np.polyfit(t, y, deg=2)  # returns [a, b, c]
            self.polys.append(p)
            self.lengths.append(len(y))
        return self

    def _distance(self, y_obs: np.ndarray, p: np.ndarray) -> float:
        t = np.arange(len(y_obs), dtype=float)
        y_hat = np.polyval(p, t)
        return float(np.mean(np.abs(y_obs - y_hat)))  # L1/n (normalized by length)

    def predict_rul_distribution(self, obs_fused: pd.DataFrame) -> Tuple[float, Tuple[float, float], np.ndarray]:
        """
        Given observed prefix for one engine, find k nearest (smallest residual distance)
        and use their (N_train - len_obs) as the neighbor RULs.
        Returns (estimated_RUL_median, (ci_low, ci_high), neighbor_RUL_array)
        """
        y_obs = obs_fused["health_indicator"].values.astype(float)
        len_obs = len(y_obs)

        dists, neighbor_ruls = [], []
        for p, n_total in zip(self.polys, self.lengths):
            d = self._distance(y_obs, p)
            dists.append(d)
            neighbor_ruls.append(max(0, n_total - len_obs))  # ensure non-negative

        dists = np.array(dists)
        neighbor_ruls = np.array(neighbor_ruls, dtype=float)

        # pick k nearest neighbors
        k_eff = min(self.k, len(dists))
        nn_idx = np.argpartition(dists, kth=k_eff - 1)[:k_eff]
        nn_ruls = neighbor_ruls[nn_idx]

        # estimate as median; 90% CI via 5th/95th percentiles (as in MATLAB visual)
        est = float(np.median(nn_ruls))
        ci = (float(np.percentile(nn_ruls, 5)), float(np.percentile(nn_ruls, 95)))
        return est, ci, nn_ruls, nn_idx, neighbor_ruls
    

    
