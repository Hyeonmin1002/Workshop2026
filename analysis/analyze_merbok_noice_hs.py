#!/usr/bin/env python3
"""
Merbok 2022 WW3 significant-wave-height diagnostics.

Input:
  analysis/noice_hs/*.nc
Output:
  analysis/noice_hs_results/
    hs_record_diagnostics.csv
    hs_valid_times.csv
    hs_domain_timeseries.png
    hs_peak_track.png
    hs_map_*.png
    hs_maps_panel.png
    hs_animation.gif
    hs_animation.mp4 (if ffmpeg is available)

The script deliberately diagnoses the mediator-output cadence before doing
physical interpretation. Records that are entirely zero are retained in the
diagnostic CSV but excluded from the "valid Hs" physical plots.
"""

from pathlib import Path
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    warnings.warn("cartopy unavailable: using plain lon/lat axes.")

ROOT = Path(__file__).resolve().parents[1]
INDIR = ROOT / "noice_hs"
OUTDIR = ROOT / "noice_hs_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

VAR = "wavImp_Sw_hs_avg"
LONVAR = "wavImp_lon"
LATVAR = "wavImp_lat"

files = sorted(INDIR.glob("*.nc"))
if not files:
    raise FileNotFoundError(f"No NetCDF files found in {INDIR}")

print(f"Reading {len(files)} files from {INDIR}")

def finite_stats(a):
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x) & (np.abs(x) < 1e20)]
    if x.size == 0:
        return dict(n=0, nonzero=0, mean=np.nan, p50=np.nan, p90=np.nan,
                    p99=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        nonzero=int(np.count_nonzero(x)),
        mean=float(np.mean(x)),
        p50=float(np.percentile(x, 50)),
        p90=float(np.percentile(x, 90)),
        p99=float(np.percentile(x, 99)),
        max=float(np.max(x)),
    )

records = []
valid_fields = []

for f in files:
    with xr.open_dataset(f, decode_times=True) as ds:
        if VAR not in ds:
            raise KeyError(f"{VAR} not found in {f.name}")
        nt = ds.sizes.get("time", 1)
        for it in range(nt):
            da = ds[VAR].isel(time=it) if "time" in ds[VAR].dims else ds[VAR]
            arr = np.asarray(da.values, dtype=float)
            arr[np.abs(arr) >= 1e20] = np.nan
            st = finite_stats(arr)
            t = pd.Timestamp(ds["time"].isel(time=it).values) if "time" in ds else pd.NaT
            records.append({
                "file": f.name, "record": it, "time": t,
                **st,
                "zero_fraction": (1.0 - st["nonzero"]/st["n"]) if st["n"] else np.nan,
            })

            # A physical field must contain at least one finite positive Hs value.
            if st["nonzero"] > 0 and np.nanmax(arr) > 0:
                lon_da = ds[LONVAR]
                lat_da = ds[LATVAR]
                if "time" in lon_da.dims:
                    lon_da = lon_da.isel(time=it)
                if "time" in lat_da.dims:
                    lat_da = lat_da.isel(time=it)
                lon = np.asarray(lon_da.values, dtype=float)
                lat = np.asarray(lat_da.values, dtype=float)

                # normalize longitudes for Alaska/Bering visualization
                lon_plot = ((lon + 180.0) % 360.0) - 180.0

                # peak location, excluding zero/fill
                physical = np.where((arr > 0) & np.isfinite(arr), arr, np.nan)
                flat_idx = int(np.nanargmax(physical))
                iy, ix = np.unravel_index(flat_idx, physical.shape)
                valid_fields.append({
                    "time": t, "hs": physical.copy(),
                    "lon": lon_plot.copy(), "lat": lat.copy(),
                    "mean": float(np.nanmean(physical)),
                    "p90": float(np.nanpercentile(physical, 90)),
                    "p99": float(np.nanpercentile(physical, 99)),
                    "max": float(np.nanmax(physical)),
                    "peak_lon": float(lon_plot[iy, ix]),
                    "peak_lat": float(lat[iy, ix]),
                })

diag = pd.DataFrame(records).sort_values("time")
diag.to_csv(OUTDIR / "hs_record_diagnostics.csv", index=False)

valid = pd.DataFrame([{
    "time": x["time"], "mean_hs_m": x["mean"], "p90_hs_m": x["p90"],
    "p99_hs_m": x["p99"], "max_hs_m": x["max"],
    "peak_lon": x["peak_lon"], "peak_lat": x["peak_lat"],
} for x in valid_fields]).sort_values("time")
valid.to_csv(OUTDIR / "hs_valid_times.csv", index=False)

print("\n=== RECORD CADENCE DIAGNOSTIC ===")
print(f"Total records : {len(diag)}")
print(f"All-zero      : {(diag.nonzero == 0).sum()}")
print(f"Nonzero       : {(diag.nonzero > 0).sum()}")
print("\nNonzero records:")
print(valid.to_string(index=False))

if not valid_fields:
    raise RuntimeError("No nonzero Hs fields found; see hs_record_diagnostics.csv")

valid_fields = sorted(valid_fields, key=lambda x: x["time"])
all_positive = np.concatenate([x["hs"][np.isfinite(x["hs"])] for x in valid_fields])
vmax = float(np.nanpercentile(all_positive, 99.5))
vmax = max(1.0, np.ceil(vmax))
print(f"Common map scale: 0 to {vmax:.1f} m (99.5th percentile)")

def make_axis(figsize=(8.2, 6.5)):
    if HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.8)
        ax.add_feature(cfeature.LAND, alpha=0.25)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.25)
    return fig, ax

def draw_field(ax, item, title=None):
    kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    mesh = ax.pcolormesh(item["lon"], item["lat"], item["hs"],
                         shading="auto", vmin=0, vmax=vmax, **kw)
    ax.plot(item["peak_lon"], item["peak_lat"], marker="*", markersize=11,
            markeredgecolor="black", markerfacecolor="white", **kw)
    if title is None:
        title = (f'{item["time"]:%Y-%m-%d %H:%M} UTC | '
                 f'max={item["max"]:.2f} m, mean={item["mean"]:.2f} m')
    ax.set_title(title)
    return mesh

# 1. Domain statistics time series
fig, ax = plt.subplots(figsize=(10, 5))
t = pd.to_datetime([x["time"] for x in valid_fields])
ax.plot(t, [x["mean"] for x in valid_fields], "o-", label="Domain mean (Hs>0)")
ax.plot(t, [x["p90"] for x in valid_fields], "o-", label="P90")
ax.plot(t, [x["p99"] for x in valid_fields], "o-", label="P99")
ax.plot(t, [x["max"] for x in valid_fields], "o-", label="Maximum")
ax.set_ylabel("Significant wave height (m)")
ax.set_title("Merbok 2022 — WW3 Hs, no-ice experiment")
ax.grid(alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTDIR / "hs_domain_timeseries.png", dpi=180)
plt.close(fig)

# 2. Individual maps
for item in valid_fields:
    fig, ax = make_axis()
    mesh = draw_field(ax, item)
    cb = fig.colorbar(mesh, ax=ax, pad=0.03, shrink=0.82)
    cb.set_label("Significant wave height, Hs (m)")
    fig.tight_layout()
    fig.savefig(OUTDIR / f'hs_map_{item["time"]:%Y%m%d_%H%M}.png',
                dpi=180, bbox_inches="tight")
    plt.close(fig)

# 3. Multi-day panel
n = len(valid_fields)
ncol = 4
nrow = int(np.ceil(n/ncol))
if HAS_CARTOPY:
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.3*nrow),
                             subplot_kw={"projection": ccrs.PlateCarree()})
else:
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.3*nrow))
axes = np.atleast_1d(axes).ravel()
last_mesh = None
for ax, item in zip(axes, valid_fields):
    if HAS_CARTOPY:
        ax.coastlines(resolution="50m", linewidth=0.6)
        ax.add_feature(cfeature.LAND, alpha=0.25)
    else:
        ax.grid(alpha=0.2)
    last_mesh = draw_field(
        ax, item,
        f'{item["time"]:%b %d %H UTC}\nmax {item["max"]:.1f} m'
    )
for ax in axes[n:]:
    ax.set_visible(False)
if last_mesh is not None:
    cb = fig.colorbar(last_mesh, ax=axes[:n].tolist(), orientation="horizontal",
                      fraction=0.035, pad=0.06)
    cb.set_label("Significant wave height, Hs (m)")
fig.suptitle("Merbok 2022 — daily nonzero WW3 Hs snapshots (NO ICE)", y=0.995)
fig.savefig(OUTDIR / "hs_maps_panel.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 4. Peak-Hs track
fig, ax = make_axis()
lons = [x["peak_lon"] for x in valid_fields]
lats = [x["peak_lat"] for x in valid_fields]
kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
ax.plot(lons, lats, "o-", **kw)
for item in valid_fields:
    ax.text(item["peak_lon"], item["peak_lat"],
            item["time"].strftime("%d"), fontsize=8, **kw)
ax.set_title("Location of domain maximum Hs (labels = September day)")
fig.tight_layout()
fig.savefig(OUTDIR / "hs_peak_track.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 5. GIF animation of valid snapshots
fig, ax = make_axis(figsize=(8.5, 6.5))
def animate(i):
    ax.clear()
    if HAS_CARTOPY:
        ax.coastlines(resolution="50m", linewidth=0.8)
        ax.add_feature(cfeature.LAND, alpha=0.25)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    item = valid_fields[i]
    mesh = draw_field(ax, item)
    return [mesh]

ani = FuncAnimation(fig, animate, frames=len(valid_fields), interval=900, blit=False)
gif_path = OUTDIR / "hs_animation.gif"
ani.save(gif_path, writer=PillowWriter(fps=1.2), dpi=130)
plt.close(fig)

# Optional MP4 from GIF if ffmpeg exists
if shutil.which("ffmpeg"):
    mp4_path = OUTDIR / "hs_animation.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(gif_path),
        "-movflags", "+faststart", "-pix_fmt", "yuv420p", str(mp4_path)
    ], check=False)

# 6. concise text report
with open(OUTDIR / "README_results.txt", "w") as fp:
    fp.write("Merbok 2022 NO-ICE Hs diagnostics\n")
    fp.write("="*40 + "\n")
    fp.write(f"Input files: {len(files)}\n")
    fp.write(f"Total hourly records: {len(diag)}\n")
    fp.write(f"All-zero records: {(diag.nonzero == 0).sum()}\n")
    fp.write(f"Nonzero records: {(diag.nonzero > 0).sum()}\n")
    fp.write(f"Valid physical snapshots: {len(valid_fields)}\n")
    peak = max(valid_fields, key=lambda x: x["max"])
    fp.write(f'Largest domain Hs: {peak["max"]:.3f} m at {peak["time"]} UTC\n')
    fp.write(f'Peak location: lon={peak["peak_lon"]:.4f}, lat={peak["peak_lat"]:.4f}\n')
    fp.write("\nIMPORTANT: all-zero mediator records are diagnosed separately and are not\n")
    fp.write("interpreted as physical calm conditions.\n")

print(f"\nDone. Results written to: {OUTDIR}")
for p in sorted(OUTDIR.iterdir()):
    print(" ", p.name)
