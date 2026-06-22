# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dry-air constants (SI units)
R_D = 287.05   # J/(kg·K)
G = 9.80665    # m/s²


class HypsometricLoss(nn.Module):
    """
    Physics-informed loss from the hypsometric equation.

    Compares the predicted geopotential-height thickness between two pressure
    levels with the thickness implied by the layer-mean temperature:

        ΔZ_theory = (R_d / g) * T_avg * ln(P_lower / P_upper)
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

        ln_p_ratio = math.log(pressure_lower / pressure_upper)
        self.register_buffer("ln_p_ratio", torch.tensor(ln_p_ratio))
        self.register_buffer("scale", torch.tensor(R_d / g))

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        z_upper = predictions[:, self.geopotential_upper_idx]
        z_lower = predictions[:, self.geopotential_lower_idx]
        predicted_thickness = z_upper - z_lower

        t_upper = predictions[:, self.temperature_upper_idx]
        t_lower = predictions[:, self.temperature_lower_idx]
        t_avg = 0.5 * (t_upper + t_lower)

        theoretical_thickness = self.scale * t_avg * self.ln_p_ratio

        return F.mse_loss(predicted_thickness, theoretical_thickness)
