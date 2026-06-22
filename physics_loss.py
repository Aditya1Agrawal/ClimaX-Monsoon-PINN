# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CLI entry point; implementation lives in climax.utils.physics_loss."""

import torch

from climax.utils.physics_loss import HypsometricLoss

__all__ = ["HypsometricLoss"]


if __name__ == "__main__":
    predictions = torch.randn(1, 5, 32, 64)

    loss_fn = HypsometricLoss(
        geopotential_upper_idx=0,
        geopotential_lower_idx=1,
        temperature_upper_idx=2,
        temperature_lower_idx=3,
        pressure_upper=50000.0,
        pressure_lower=85000.0,
    )

    print(loss_fn(predictions).item())
