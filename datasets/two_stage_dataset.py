import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np, glob, os
from datetime import datetime, timedelta
import pandas as pd
import json
import time
from .cal_clear_ghi import compute_clearsky_ineichen_np

class TwoStageDataset(Dataset):
    def __init__(self,
                 data_dir,
                 latlon_path,
                 stats=None,
                 era5_dir=None,
                 baguan_dir=None,
                 era5_stats=None,
                 history_frames=6,
                 future_frames=24,
                 target_size=512,
                 split='train'):
        self.history_frames = history_frames
        self.future_frames  = future_frames
        self.seq_len = history_frames + future_frames
        self.target_size = target_size

        self.era5_dir = era5_dir
        self.baguan_dir = baguan_dir

        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        era5_files = sorted(glob.glob(os.path.join(era5_dir, "*.npy"))) if era5_dir else []
        self.era5_files = {os.path.basename(f): f for f in era5_files}

        baguan_files = sorted(glob.glob(os.path.join(baguan_dir, "*.npy"))) if baguan_dir else []
        self.baguan_files = {os.path.basename(f): f for f in baguan_files}

        if stats:
            self.cmin = torch.tensor(stats['mins']).view(6, 1, 1).float()
            self.cmax = torch.tensor(stats['maxs']).view(6, 1, 1).float()
        else:
            self.cmin = torch.zeros(1, 6, 1, 1)
            self.cmax = torch.ones(1, 6, 1, 1)

        if era5_stats is None:
            raise ValueError("era5_stats must be provided for normalization.")

        if ("mins" in era5_stats) and ("maxs" in era5_stats):
            self.era5_mins = torch.tensor(era5_stats["mins"], dtype=torch.float32).view(-1, 1, 1)
            self.era5_maxs = torch.tensor(era5_stats["maxs"], dtype=torch.float32).view(-1, 1, 1)
            self.era5_use_minmax = True
        elif ("mean" in era5_stats) and ("std" in era5_stats):
            self.era5_mean = torch.tensor(era5_stats["mean"], dtype=torch.float32).view(-1, 1, 1)
            self.era5_std  = torch.tensor(era5_stats["std"], dtype=torch.float32).view(-1, 1, 1)
            self.era5_use_minmax = False
        else:
            raise ValueError("Unsupported ERA5 stats json format. Need mins/maxs or mean/std.")

        self.valid_sequences = []
        for i in range(len(self.files) - self.seq_len + 1):
            seq = self.files[i:i+self.seq_len]
            
            seq_start = seq[history_frames-1].split('/')[-1]
            mm = seq_start[4:6]
            hh = seq_start[9:11]
            # if (mm < '07') or (hh not in ["00", "12"]):
            if (hh not in ["00", "12"]):
                continue

            seq_start = seq[history_frames-1].split('/')[-1]
            mm = seq_start[4:6]
            hh = seq_start[9:11]
            if self._check_continuous(seq):
                ok = True
                for sf in seq:
                    bn = os.path.basename(sf)
                    if bn not in self.era5_files:
                        ok = False
                        break
                if not ok:
                    continue
            self.valid_sequences.append(seq)

        split_idx = int(len(self.valid_sequences) * 0.9)
        if split == 'train':
            self.valid_sequences = self.valid_sequences[:split_idx]
        elif split == 'val':
            self.valid_sequences = self.valid_sequences[split_idx:]
        elif split == 'test':
            self.valid_sequences = self.valid_sequences
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

        self.latlon_map = np.load(latlon_path)
        self.lat_arr = self.latlon_map[...,0]
        self.lon_arr = self.latlon_map[...,1]
        self.altitude = 10.0

    def _check_continuous(self, file_list):
        """ 检查时间连续性 """
        times=[]
        for f in file_list:
            try:
                t = datetime.strptime(os.path.basename(f).replace(".npy",""), "%Y%m%d_%H%M")
                times.append(t)
            except:
                return False
        return all((times[i]-times[i-1])==timedelta(hours=1) for i in range(1,len(times)))

    def normalize(self, data, time_point_utc):
        rng = self.cmax - self.cmin
        rng = torch.where(rng == 0, torch.ones_like(rng), rng)
        normed = 2 * (data - self.cmin) / rng - 1

        ghi_clear_full = compute_clearsky_ineichen_np(
            self.lat_arr, self.lon_arr, self.altitude, time_point_utc
        ).astype(np.float32)

        ghi_clear_resized = F.interpolate(
            torch.from_numpy(ghi_clear_full).unsqueeze(0).unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0)

        epsilon = 10.0
        mask = ghi_clear_resized > epsilon
        ghi_orig = data[0]
        ghi_ratio = torch.zeros_like(ghi_orig, dtype=torch.float32)
        ghi_ratio[mask] = ghi_orig[mask] / ghi_clear_resized[mask]
        ghi_ratio = torch.clamp(ghi_ratio, 0.0, 1.0)
        ghi_ratio[~mask] = 0.0

        normed = torch.cat([normed, ghi_ratio.unsqueeze(0)], dim=0)
        return normed

    def normalize_era5(self, era5):
        if self.era5_use_minmax:
            rng = self.era5_maxs - self.era5_mins
            rng = torch.where(rng == 0, torch.ones_like(rng), rng)
            out = 2 * (era5 - self.era5_mins) / rng - 1
        else:
            std = torch.where(self.era5_std == 0, torch.ones_like(self.era5_std), self.era5_std)
            out = (era5 - self.era5_mean) / std
        return out

    def __getitem__(self, idx):
        seq_files = self.valid_sequences[idx]

        frames=[]; times=[]
        era5_frames = []
        for f in seq_files:
            arr = np.load(f)   # (6,512,512)
            t = torch.tensor(arr).float()
            t = torch.nan_to_num(t)
            t = F.interpolate(t.unsqueeze(0),
                              size=(self.target_size,self.target_size),
                              mode='bilinear',
                              align_corners=False).squeeze(0)

            fname = os.path.basename(f).replace(".npy","")
            t_utc = pd.Timestamp(datetime.strptime(fname, "%Y%m%d_%H%M"), tz='UTC')

            t = self.normalize(t, t_utc)

            frames.append(t)
            times.append(fname)

            if self.era5_dir is not None:
                era5_path = self.era5_files.get(os.path.basename(f), None)
                if era5_path is None:
                    era5 = torch.zeros(41, 103, 103, dtype=torch.float32)
                else:
                    era5 = torch.tensor(np.load(era5_path)).float()
                    era5 = self.normalize_era5(torch.nan_to_num(era5))
                era5_frames.append(era5)

        if self.baguan_dir is not None:
            seq_time = seq_files[self.history_frames-1].split('/')[-1].replace("_", "")
            baguan_seq = torch.tensor(np.load(os.path.join(self.baguan_dir, seq_time))).float()
            baguan_seq = torch.nan_to_num(baguan_seq)
            baguan_seq[:, 35] = baguan_seq[:, 35] / 100.
            baguan_seq[:, 36] = baguan_seq[:, 36] / 100.
            baguan_seq = self.normalize_era5(baguan_seq)

        data = torch.stack(frames, dim=0) 
        his_satellite   = data[:self.history_frames, 2:6]
        fut_cloud_label = data[self.history_frames:, 1:2]
        fut_sat_label   = data[self.history_frames:, 2:6]
        fut_ghi_label   = data[self.history_frames:, 0:1]

        fut_clearghi=[]
        fut_leadtime=[]
        fut_ratio_label=[]
        for i, fname in enumerate(times[self.history_frames:]):
            t_utc = pd.Timestamp(datetime.strptime(fname, "%Y%m%d_%H%M"), tz='UTC')

            ghi_clear_full = compute_clearsky_ineichen_np(self.lat_arr, self.lon_arr, self.altitude, t_utc)
            ghi_clear_resized = F.interpolate(
                torch.tensor(ghi_clear_full).unsqueeze(0).unsqueeze(0).float(),
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

            fut_ratio_label.append(data[self.history_frames+i, -1].unsqueeze(0))

            fut_clearghi.append((ghi_clear_resized/1500))
            lead = torch.full_like(ghi_clear_resized, fill_value=(i+1)/self.future_frames)
            fut_leadtime.append(lead)

        fut_clearghi    = torch.stack(fut_clearghi, dim=0)
        fut_leadtime    = torch.stack(fut_leadtime, dim=0)
        fut_ratio_label = torch.stack(fut_ratio_label, dim=0)

        event_name = f"S{times[0]}_E{times[-1]}"
        out = dict({
            "event_name": event_name,
            "his_satellite": his_satellite,
            "fut_cloud_label": fut_cloud_label,
            "fut_satellite": fut_sat_label,
            "fut_clearghi": fut_clearghi,
            "fut_leadtime": fut_leadtime,
            "fut_ghi_label": fut_ghi_label,
            "fut_ratio_label": fut_ratio_label,
            "times": times,
        })

        if self.era5_dir is not None:
            era5_seq = torch.stack(era5_frames, dim=0)  # [seq_len,41,103,103]
            era5_seq = F.interpolate(
                era5_seq,
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )
            out["his_era5"] = era5_seq[:self.history_frames]
            out["fut_era5"] = era5_seq[self.history_frames:]
            if self.baguan_dir is not None:
                baguan_seq = F.interpolate(
                    baguan_seq,
                    size=(self.target_size, self.target_size),
                    mode='bilinear', align_corners=False
                )
                out["fut_baguan"] = baguan_seq

        return out

    def __len__(self):
        return len(self.valid_sequences)

if __name__ == "__main__":
    # 原卫星统计
    stats_path = "./datasets/data_train_statistics.json"
    with open(stats_path, "r") as f:
        stats = json.load(f)
    era_stats_path = "./datasets/modify_era5_train.json"
    with open(era_stats_path, "r") as f:
        era_stats = json.load(f)

    dataset = TwoStageDataset(
        data_dir="./datasets/data/test",
        latlon_path="./datasets/latlon_512x512.npy",
        stats=stats,
        era5_dir="./datasets/data/era5_test",
        baguan_dir='./datasets/data/baguan_test_2025',
        era5_stats=era_stats,
        history_frames=6,
        future_frames=24,
        target_size=512,
        split="test",
    )

    print("样本数:", len(dataset))

    sample = dataset[0]
    print("event_name:", sample["event_name"])

    # 卫星（原输出）
    print("his_satellite:", sample["his_satellite"].shape,
          "min/max:", sample["his_satellite"].min().item(), sample["his_satellite"].max().item())
    print("fut_cloud_label:", sample["fut_cloud_label"].shape,
          "min/max:", sample["fut_cloud_label"].min().item(), sample["fut_cloud_label"].max().item())
    print("fut_satellite:", sample["fut_satellite"].shape,
          "min/max:", sample["fut_satellite"].min().item(), sample["fut_satellite"].max().item())
    print("fut_clearghi:", sample["fut_clearghi"].shape,
          "min/max:", sample["fut_clearghi"].min().item(), sample["fut_clearghi"].max().item())
    print("fut_leadtime:", sample["fut_leadtime"].shape,
          "min/max:", sample["fut_leadtime"].min().item(), sample["fut_leadtime"].max().item())
    print("fut_ghi_label:", sample["fut_ghi_label"].shape,
          "min/max:", sample["fut_ghi_label"].min().item(), sample["fut_ghi_label"].max().item())
    print("fut_ratio_label:", sample["fut_ratio_label"].shape,
          "min/max:", sample["fut_ratio_label"].min().item(), sample["fut_ratio_label"].max().item())

    # ERA5（新增输出）
    print("his_era5:", sample["his_era5"].shape,
          "min/max:", sample["his_era5"].min().item(), sample["his_era5"].max().item())
    print("fut_era5:", sample["fut_era5"].shape,
          "min/max:", sample["fut_era5"].min().item(), sample["fut_era5"].max().item())
    print("fut_baguan:", sample["fut_baguan"].shape,
          "min/max:", sample["fut_baguan"].min().item(), sample["fut_baguan"].max().item())
