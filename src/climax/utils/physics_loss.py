# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dry-air constants (SI units)
R_D = 287.05      # J/(kg·K)
G = 9.80665       # m/s²


class HypsometricLoss(nn.Module):
    """
    Physics-informed Hypsometric Loss.

    Uses the residual form

        R = ΔZ - (Rd/g) * T_avg * ln(P_lower/P_upper)

    and minimizes

        L = MSE(R_pred, R_target)

    where

        ΔZ = (Φ_upper - Φ_lower) / g

    because ERA5 stores geopotential Φ (m²/s²), not
    geopotential height Z (m).
    """

    def __init__(
        self,
        geopotential_upper_idx: int,
        geopotential_lower_idx: int,
        temperature_upper_idx: int,
        temperature_lower_idx: int,
        pressure_upper: float,
        pressure_lower: float,
        R_d: float = R_D,
        g: float = G,
    ):
        super().__init__()

        self.geopotential_upper_idx = geopotential_upper_idx
        self.geopotential_lower_idx = geopotential_lower_idx
        self.temperature_upper_idx = temperature_upper_idx
        self.temperature_lower_idx = temperature_lower_idx

        self.register_buffer(
            "ln_p_ratio",
            torch.tensor(math.log(pressure_lower / pressure_upper), dtype=torch.float32),
        )

        self.register_buffer(
            "scale",
            torch.tensor(R_d / g, dtype=torch.float32),
        )

        self.register_buffer(
            "g",
            torch.tensor(g, dtype=torch.float32),
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Prediction variables
        # --------------------------------------------------

        phi_upper_pred = predictions[:, self.geopotential_upper_idx]
        phi_lower_pred = predictions[:, self.geopotential_lower_idx]

        # ERA5 geopotential -> geopotential height
        thickness_pred = (phi_upper_pred - phi_lower_pred) / self.g

        t_upper_pred = predictions[:, self.temperature_upper_idx]
        t_lower_pred = predictions[:, self.temperature_lower_idx]

        t_avg_pred = 0.5 * (t_upper_pred + t_lower_pred)

        thickness_theory_pred = (
            self.scale * t_avg_pred * self.ln_p_ratio
        )

        residual_pred = thickness_pred - thickness_theory_pred

        # --------------------------------------------------
        # Target variables
        # --------------------------------------------------

        phi_upper_target = targets[:, self.geopotential_upper_idx]
        phi_lower_target = targets[:, self.geopotential_lower_idx]

        thickness_target = (phi_upper_target - phi_lower_target) / self.g

        t_upper_target = targets[:, self.temperature_upper_idx]
        t_lower_target = targets[:, self.temperature_lower_idx]

        t_avg_target = 0.5 * (t_upper_target + t_lower_target)

        thickness_theory_target = (
            self.scale * t_avg_target * self.ln_p_ratio
        )

        residual_target = (
            thickness_target - thickness_theory_target
        )

        # --------------------------------------------------
        # Physics-informed loss
        # --------------------------------------------------

        return F.mse_loss(residual_pred, residual_target)