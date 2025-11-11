import sys
sys.path.insert(0, "/xace/d1/yashd1/aurora_temp_ins")

import aurora
print(f"✅ Aurora package is from: {aurora.__file__}")

import torch
import xarray as xr
import numpy as np
from pathlib import Path
from aurora import Aurora, rollout, Batch, Metadata

# === CONFIG ===
DATA_DIR = Path("/xace/d1")
OUTFILE = Path("/xace/d1/yashd1/auroraoutput_temp_inserted.nc")
STEPS = 240  # 60 days × 4 steps/day

# === LOAD ERA5 FILES ===
static_ds = xr.open_dataset(DATA_DIR / "static.nc", engine="netcdf4")
surf_ds = xr.open_dataset(DATA_DIR / "2023-02-01-surface-level.nc", engine="netcdf4")
atmos_ds = xr.open_dataset(DATA_DIR / "2023-02-01-atmospheric.nc", engine="netcdf4")

# === LOAD TRUTH FOR INJECTION & RMSE PRINTS ===
truth_ds = xr.open_dataset("/xace/d1/yashd1/era5_temp_aurora.grib", engine="cfgrib")

# === PREPARE BATCH ===
batch = Batch(
    surf_vars={
        "2t": torch.from_numpy(surf_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_ds["msl"].values[:2][None]),
    },
    static_vars={
        "z": torch.from_numpy(static_ds["z"].values[0]),
        "slt": torch.from_numpy(static_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_ds.latitude.values),
        lon=torch.from_numpy(surf_ds.longitude.values),
        time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(p) for p in atmos_ds.pressure_level.values),
    ),
)
print("✅ Batch created (official style)")

# === LOAD MODEL ===
model = Aurora(use_lora=False)
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval()
print("🧠 Aurora model loaded")

# === RUN INFERENCE WITH RMSE MONITORING ===
print(f"🚀 Running {STEPS}-step inference...")
with torch.inference_mode():
    preds = [
        pred.to("cpu")
        for pred in rollout(
            model,
            batch,
            steps=STEPS,
            truth_ds=truth_ds,
            start_inject_step=121  # 🔁 truth injection begins here
        )
    ]
print("✅ Forecast complete")
# === FILTER TO ONLY 00:00 and 12:00 UTC FORECASTS ===
preds_filtered = []
lead_times = []

for i in range(STEPS):
    hour = (6 * i) % 24
    if hour in [0, 12]:
        preds_filtered.append(preds[i])
        lead_times.append(6 * i)

print(f"💾 Saving {len(preds_filtered)} steps at 00:00 and 12:00 UTC only")

# === COLLECT VARIABLES ===
lat = preds[0].metadata.lat.numpy()
lon = preds[0].metadata.lon.numpy()
levels = preds[0].metadata.atmos_levels
base_time = preds[0].metadata.time[0]
data_vars = {}

for key in preds[0].surf_vars.keys():
    print(f"📦 Saving surface var: {key}")
    stack = [pred.surf_vars[key][0, 0].numpy() for pred in preds_filtered]
    data_vars[key] = (["lead_time", "lat", "lon"], np.stack(stack))

for key in preds[0].atmos_vars.keys():
    print(f"📦 Saving atmospheric var: {key}")
    for i, level in enumerate(levels):
        stack = [pred.atmos_vars[key][0, 0, i].numpy() for pred in preds_filtered]
        data_vars[f"{key}{level}"] = (["lead_time", "lat", "lon"], np.stack(stack))

# === SAVE NETCDF ===
ds = xr.Dataset(
    data_vars,
    coords={
        "time": [np.datetime64(base_time)],
        "lead_time": lead_times,
        "lat": lat,
        "lon": lon,
    },
)
ds.to_netcdf(OUTFILE)
print(f"✅ Saved filtered forecast to: {OUTFILE}")
