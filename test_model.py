# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import types

import numpy as np
import torch

# ClimaX pos_embed uses deprecated np.float (removed in NumPy 2.0)
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

from climax.arch import ClimaX

# Default variables and architecture for 5.625° resolution (img_size [32, 64])
# from configs/pretrain_climax.yaml
DEFAULT_VARS = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential_50",
    "geopotential_250",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "u_component_of_wind_50",
    "u_component_of_wind_250",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "v_component_of_wind_50",
    "v_component_of_wind_250",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "temperature_50",
    "temperature_250",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "relative_humidity_50",
    "relative_humidity_250",
    "relative_humidity_500",
    "relative_humidity_600",
    "relative_humidity_700",
    "relative_humidity_850",
    "relative_humidity_925",
    "specific_humidity_50",
    "specific_humidity_250",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
]


def main():
    model = ClimaX(
        default_vars=DEFAULT_VARS,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
    )
    model.eval()
    model.cpu()

    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars], dtype=np.int64)
        return torch.from_numpy(ids).to(device)

    model.get_var_ids = types.MethodType(get_var_ids, model)

    x = torch.randn(1, 48, 32, 64)
    y = torch.randn(1, 48, 32, 64)
    lead_times = torch.zeros(1)

    with torch.no_grad():
        _, output = model.forward(
            x,
            y,
            lead_times,
            DEFAULT_VARS,
            DEFAULT_VARS,
            metric=None,
            lat=None,
        )

    print(tuple(output.shape))


if __name__ == "__main__":
    main()
