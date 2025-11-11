import dataclasses
from typing import Generator

import torch
from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

def rollout(
    model: Aurora,
    batch: Batch,
    steps: int,
    truth_ds=None,
    start_inject_step: int = 121
) -> Generator[Batch, None, None]:
    """Perform a roll-out with optional truth temperature injection.

    Args:
        model (Aurora): The model to roll out.
        batch (Batch): Initial batch.
        steps (int): Number of steps.
        truth_ds (xarray.Dataset, optional): ERA5 dataset with variable 't' (13 pressure levels).
        start_inject_step (int): Step to begin truth injection.

    Yields:
        Batch: Forecast at each step.
    """
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # Pressure level → index lookup
    level_map = {50: 0, 100: 1, 150: 2, 200: 3, 250: 4, 300: 5, 400: 6,
                 500: 7, 600: 8, 700: 9, 850: 10, 925: 11, 1000: 12}

    for step in range(steps):
        pred = model.forward(batch)

        # 🔍 Debug info at start of injection
        if step == start_inject_step:
            print(f"\n🚨 Step {step}: Starting truth injection")
            print("🔍 pred.atmos_vars.keys():", list(pred.atmos_vars.keys()))
            if truth_ds is not None:
                print("🔍 truth_ds.keys():", list(truth_ds.keys()))
            else:
                raise RuntimeError("❌ truth_ds is None!")

        if truth_ds is not None and step >= start_inject_step:
            if "t" not in pred.atmos_vars:
                raise KeyError(f"❌ 't' not found in pred.atmos_vars at step {step}")
            if "t" not in truth_ds:
                raise KeyError(f"❌ 't' not found in truth_ds at step {step}")

            # ✅ Use forecast time for exact time match
            forecast_time = pred.metadata.time[0]
            try:
                truth_slice = truth_ds.sel(time=forecast_time, method="nearest")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to match forecast time {forecast_time} in truth_ds: {e}")

            # ✅ Fix shape mismatch: slice latitude and add batch+time dims
            truth_tensor = torch.from_numpy(
                truth_slice["t"].isel(latitude=slice(0, 720)).values
            ).to(p.device).type(p.dtype)  # shape: (13, 720, 1440)

            truth_tensor = truth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 13, 720, 1440)

            if truth_tensor.shape != pred.atmos_vars["t"].shape:
                raise ValueError(f"❌ Shape mismatch at step {step}: truth {truth_tensor.shape} vs pred {pred.atmos_vars['t'].shape}")

            print(f"✅ Injecting truth for 't' at step {step} | shape: {truth_tensor.shape}")
            # 📏 RMSE before injection
            #for level in [500, 1000]:
             #   idx = level_map[level]
              #  rmse_before = torch.sqrt(torch.mean((pred.atmos_vars["t"][:, :, idx] - truth_tensor[:, :, idx]) ** 2))
               # print(f"Step {step} | t{level} RMSE before injection: {rmse_before.item():.4f}")

            # 🧠 Inject
            pred.atmos_vars["t"] = truth_tensor

            # 📏 RMSE after injection
            #for level in [500, 1000]:
             #   idx = level_map[level]
              #  rmse_after = torch.sqrt(torch.mean((pred.atmos_vars["t"][:, :, idx] - truth_tensor[:, :, idx]) ** 2))
               # print(f"Step {step} | t{level} RMSE after injection: {rmse_after.item():.4f}")

        yield pred

        # Update batch with current prediction
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
