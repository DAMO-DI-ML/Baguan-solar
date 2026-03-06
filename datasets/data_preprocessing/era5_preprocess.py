import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

src_root = "./data/npy_era5"
out_root = "./data/era5_test"
stats_json = "./data/era5_channels_stats_train.json"

os.makedirs(out_root, exist_ok=True)

levels = [925, 850, 700, 600, 500, 250, 50]

lon_min, lon_max = 101.20, 126.80
lat_min, lat_max = 21.20, 46.80
grid_ref_grib = "./data/grib_era5/grib_1h/low_cloud_cover/low_cloud_cover_2021.grib"

channels = []
for p in levels:
    channels.append((f"z_{p}", f"z_{p}"))
for p in levels:
    channels.append((f"q_{p}", f"q_{p}"))
for p in levels:
    channels.append((f"t_{p}", f"t_{p}"))
for p in levels:
    channels.append((f"u_{p}", f"u_{p}"))
for p in levels:
    channels.append((f"v_{p}", f"v_{p}"))

channels += [
    ("lcc", "lcc"),
    ("tcc", "tcc"),
    ("tcw", "tcw"),
    ("tcwv", "tcwv"),
    ("avg_sdswrf", "avg_sdswrf"),
    ("avg_sdirswrf", "avg_sdirswrf"),
]

channel_names = [c[0] for c in channels]

ref_folder = os.path.join(src_root, channels[0][1])
ref_files = glob(os.path.join(ref_folder, "*", "*", "*.npy"))
timestamps = sorted(os.path.splitext(os.path.basename(f))[0] for f in ref_files)
# 只处理 2021-2024
timestamps = [ts for ts in timestamps if ts[:6] >= "2021" and ts[:6] <= "2024"]


# -------------------- compute strict crop slices from lat/lon --------------------
def _normalize_lon_0_360(lon):
    lon = np.asarray(lon, dtype=np.float64)
    return (lon + 360.0) % 360.0

def compute_strict_slices_from_grib(grib_path, lon_min, lon_max, lat_min, lat_max):
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "/tmp/cfgrib-{short_hash}.idx"},
    )
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError(f"GRIB中未找到 latitude/longitude，coords={list(ds.coords)}")

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    lon360 = _normalize_lon_0_360(lon)
    lon_min360 = float(_normalize_lon_0_360(lon_min))
    lon_max360 = float(_normalize_lon_0_360(lon_max))

    lat_desc = (lat[0] > lat[-1])

    if lat_desc:
        y0 = np.searchsorted((-lat), (-lat_max), side="left")
        y1 = np.searchsorted((-lat), (-lat_min), side="right")
    else:
        y0 = np.searchsorted(lat, lat_min, side="left")
        y1 = np.searchsorted(lat, lat_max, side="right")

    if lon_min360 <= lon_max360:
        x0 = np.searchsorted(lon360, lon_min360, side="left")
        x1 = np.searchsorted(lon360, lon_max360, side="right")
    else:
        raise ValueError("lon范围跨越0/360经线，当前严格裁剪代码未实现拼接裁剪。")

    if not (0 <= y0 < y1 <= len(lat)) or not (0 <= x0 < x1 <= len(lon)):
        raise ValueError(f"裁剪slice非法: y({y0},{y1}) x({x0},{x1})")

    return int(y0), int(y1), int(x0), int(x1), lat, lon

y0, y1, x0, x1, lat_ref, lon_ref = compute_strict_slices_from_grib(
    grid_ref_grib, lon_min, lon_max, lat_min, lat_max
)

crop_lat = lat_ref[y0:y1]
crop_lon = lon_ref[x0:x1]
crop_hw = (len(crop_lat), len(crop_lon))
print("Strict crop slices:", (y0, y1, x0, x1), "=>", crop_hw)

# -------------------- stats管理 with thread-safety --------------------
stats = {
    "channels": {name: {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None} for name in channel_names},
    "dtype_saved": "float32",
}
stats_lock = Lock()

def update_stats(d, arr: np.ndarray):
    a = arr.astype(np.float64, copy=False)
    d["count"] += a.size
    d["sum"] += float(a.sum())
    d["sumsq"] += float((a * a).sum())
    amin = float(a.min())
    amax = float(a.max())
    d["min"] = amin if d["min"] is None else min(d["min"], amin)
    d["max"] = amax if d["max"] is None else max(d["max"], amax)

def finalize(d):
    if d["count"] == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "count": 0}
    mean = d["sum"] / d["count"]
    var = d["sumsq"] / d["count"] - mean * mean
    var = max(var, 0.0)
    return {"mean": mean, "std": var ** 0.5, "min": d["min"], "max": d["max"], "count": d["count"]}

# -------------------- 单个timestamp的处理函数 --------------------
def process_timestamp(ts):
    """
    处理单个timestamp，返回 (success, ts, error_msg)
    """
    try:
        y = ts[:4]
        ymd = ts[:8]
        arrays = []
        missing = False

        for ch_name, folder in channels:
            path = os.path.join(src_root, folder, y, ymd, f"{ts}.npy")
            if not os.path.exists(path):
                missing = True
                break
            arr = np.load(path)  # (lat, lon)
            # arrays.append(arr[y0:y1, x0:x1])
            arrays.append(arr)

        if missing:
            return (False, ts, "missing_file")

        # validate shapes consistent
        shape0 = arrays[0].shape
        if any(a.shape != shape0 for a in arrays):
            return (False, ts, "shape_mismatch")

        stacked = np.stack(arrays, axis=0).astype(np.float32)  # (C, H, W)

        # save
        out_path = os.path.join(out_root, y)
        os.makedirs(out_path, exist_ok=True)
        out_path = os.path.join(out_path, f"{ts}.npy")
        np.save(out_path, stacked)

        # update stats (thread-safe)
        with stats_lock:
            for (ch_name, _), arr in zip(channels, arrays):
                update_stats(stats["channels"][ch_name], arr)

        return (True, ts, None)

    except Exception as e:
        return (False, ts, str(e))

# -------------------- main loop with threading --------------------
num_workers = 4  # 可根据CPU核数调整
max_workers = min(num_workers, len(timestamps))

print(f"Processing {len(timestamps)} timestamps with {max_workers} workers...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    futures = {executor.submit(process_timestamp, ts): ts for ts in timestamps}
    
    # 使用tqdm跟踪进度
    with tqdm(total=len(futures), desc="Processing") as pbar:
        for future in as_completed(futures):
            ts = futures[future]
            try:
                success, ts_result, error_msg = future.result()
            except Exception as e:
                print(f"Error in {ts}: {str(e)}")
            
            pbar.update(1)

# # -------------------- write final json --------------------
# out_stats = {
#     "per_channel": {name: finalize(stats["channels"][name]) for name in channel_names},
#     "dtype_saved": stats["dtype_saved"],
# }

# with open(stats_json, "w", encoding="utf-8") as f:
#     json.dump(out_stats, f, ensure_ascii=False, indent=2)

