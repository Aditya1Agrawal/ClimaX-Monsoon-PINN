# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, List

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from climax.arch import ClimaX
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
)
from climax.utils.physics_loss import HypsometricLoss
from climax.utils.pos_embed import interpolate_pos_embed


class GlobalForecastModule(LightningModule):
    """Lightning module for global forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: ClimaX,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        physics_lambda: float = 1e-3,
        physics_geopotential_upper: str = "geopotential_500",
        physics_geopotential_lower: str = "geopotential_850",
        physics_temperature_upper: str = "temperature_500",
        physics_temperature_lower: str = "temperature_850",
        physics_pressure_upper: float = 50000.0,
        physics_pressure_lower: float = 85000.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.hypsometric_loss = HypsometricLoss(
            geopotential_upper_idx=0,
            geopotential_lower_idx=1,
            temperature_upper_idx=2,
            temperature_lower_idx=3,
            pressure_upper=physics_pressure_upper,
            pressure_lower=physics_pressure_lower,
        )
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def _lat_weighted_mse_train(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        error = (pred - y) ** 2
        w_lat = np.cos(np.deg2rad(self.lat))
        w_lat = w_lat / w_lat.mean()
        w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)
        return (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.denormalization(x)

    def _set_physics_channel_indices(self, out_variables: List[str]) -> bool:
        var_names = (
            self.hparams.physics_geopotential_upper,
            self.hparams.physics_geopotential_lower,
            self.hparams.physics_temperature_upper,
            self.hparams.physics_temperature_lower,
        )
        try:
            indices = [out_variables.index(name) for name in var_names]
        except ValueError:
            return False

        (
            self.hypsometric_loss.geopotential_upper_idx,
            self.hypsometric_loss.geopotential_lower_idx,
            self.hypsometric_loss.temperature_upper_idx,
            self.hypsometric_loss.temperature_lower_idx,
        ) = indices
        return True

    def _compute_physics_loss(
        self,
        preds: torch.Tensor,
        y: torch.Tensor,
        out_variables: List[str],
    ) -> torch.Tensor:
        if not self._set_physics_channel_indices(out_variables):
            return preds.new_zeros(())

        preds_denorm = self._denormalize(preds)
        y_denorm = self._denormalize(y)

        return self.hypsometric_loss(
            preds_denorm,
            y_denorm,
        )
    def training_step(self, batch: Any, batch_idx: int):

        # Print only once
        if self.current_epoch == 0 and batch_idx == 0:
            print(f"physics_lambda = {self.hparams.physics_lambda}")

        x, y, lead_times, variables, out_variables = batch

        loss_dict, preds = self.net.forward(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            [lat_weighted_mse],
            lat=self.lat,
        )

        loss_dict = loss_dict[0]

        for var in loss_dict.keys():
            if var == "loss":
                continue

            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        # Forecast loss
        mse_loss = self._lat_weighted_mse_train(preds, y)

        # Physics loss
        physics_loss = self._compute_physics_loss(
            preds,
            y,
            out_variables,
        )

        # Weighted physics loss
        weighted_physics_loss = (
            self.hparams.physics_lambda * physics_loss
        )

        # Total loss
        total_loss = mse_loss + weighted_physics_loss

        # Logging
        self.log(
            "train/mse_loss",
            mse_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log(
            "train/physics_loss",
            physics_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        self.log(
            "train/weighted_physics_loss",
            weighted_physics_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch: Any, batch_idx: int):
       
        
   
        if batch_idx == 0:
            print("=" * 60)
            print(f"VALIDATION STARTED - Epoch {self.current_epoch}")
            print("=" * 60)
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=self.denormalization,
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
