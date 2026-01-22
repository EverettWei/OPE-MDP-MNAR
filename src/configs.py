'''
Configuration objects for the simulation project.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    '''
    Configuration for the MNARMDP environment.

    Parameters
    ----------
    horizon : int, default=20
        Episode length T (must be ≥ 1).
    sigma_s : float, default=0.1
        Std dev for Gaussian transition noise on S_{t+1}.
    sigma_r : float, default=0.1
        Std dev for Gaussian reward noise on R_t.
    init_mean : np.ndarray, shape (2,), default=zeros(2)
        Mean of the initial state S_1.
    init_std : float, default=1.0
        Std dev (isotropic) for the initial state S_1.
    seed : Optional[int], default=42
        RNG seed for reproducibility.
    gamma : float, default=1.0
        Discount factor for future rewards.

    Notes
    -----
    - The dataclass is frozen to avoid accidental mutation during experiments.
    - `init_mean` uses a default_factory to avoid shared mutable defaults.
    '''

    horizon: int = 20
    sigma_s: float = 0.1
    sigma_r: float = 0.1
    init_mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    init_std: float = 1.0
    seed: Optional[int] = 42
    gamma: float = 1.0

    def __post_init__(self) -> None:
        # Basic validation on shapes and ranges.
        if self.init_mean.shape != (2,):
            raise ValueError(f"init_mean must have shape (2,), got {self.init_mean.shape}.")
        if not (self.horizon >= 1):
            raise ValueError("horizon must be ≥ 1.")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0,1].")
        if not (self.sigma_s >= 0.0 and self.sigma_r >= 0.0):
            raise ValueError("sigma_s and sigma_r must be non-negative.")