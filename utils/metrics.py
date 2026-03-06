import os, csv
import torch
from typing import List, Optional, Dict

class RMSEMetrics:
    """
    在 update() 内计算每个 event_time 的 rmse_t（每个leadtime一个RMSE），并保存为表。
    - 对同一个 event_time 多次出现：对 rmse_t 做 running mean
    """

    def __init__(self, num_frames=24, device="cpu"):
        self.num_frames = num_frames
        self.device = device
        self.reset()

    def reset(self):
        self.event_cloud_rmse_sum = {}  # event -> sum_rmse_t[T]
        self.event_ghi_rmse_sum   = {}  # event -> sum_rmse_t[T]
        self.event_cnt            = {}  # event -> number of times aggregated (int)

    def _ensure_event(self, event: str):
        if event not in self.event_cnt:
            z = torch.zeros(self.num_frames, device=self.device, dtype=torch.float64)
            self.event_cloud_rmse_sum[event] = z.clone()
            self.event_ghi_rmse_sum[event]   = z.clone()
            self.event_cnt[event] = 0

    @torch.no_grad()
    def update(self,
               pred_cloud: torch.Tensor,     # [B,T,H,W]
               target_cloud: torch.Tensor,   # [B,T,H,W]
               pred_ghi: torch.Tensor,       # [B,T,H,W]
               target_ghi: torch.Tensor,     # [B,T,H,W]
               time_list: Optional[List[List[str]]] = None,
               mask: Optional[torch.Tensor] = None,          # [B,T,H,W] or [B,T,1,H,W]
               loss_value: Optional[float] = None):

        if time_list is None:
            raise ValueError("Need time_list to map each sample to event_time (e.g. time_list[b][-1]).")

        if mask is not None and mask.dim() == 5:
            mask = mask.squeeze(2)  # [B,T,H,W]

        B, T, H, W = pred_cloud.shape
        assert T == self.num_frames, f"T={T} != num_frames={self.num_frames}"

        # 用 float64 计算更稳定
        pred_cloud = pred_cloud.to(torch.float64)
        target_cloud = target_cloud.to(torch.float64)
        pred_ghi = pred_ghi.to(torch.float64)
        target_ghi = target_ghi.to(torch.float64)
        if mask is not None:
            mask = mask.to(torch.float64)

        for b in range(B):
            event_time = time_list[5][b]   # 起报时刻：history最后一帧
            self._ensure_event(event_time)

            # ---- Cloud RMSE_t: 先对 (H,W) 求 MSE，再 sqrt -> RMSE ----
            se_cloud = (pred_cloud[b] - target_cloud[b]) ** 2          # [T,H,W]
            cloud_mse_t = se_cloud.mean(dim=(1, 2))                    # [T]
            cloud_rmse_t = torch.sqrt(cloud_mse_t)                     # [T]

            # ---- GHI RMSE_t (masked optional) ----
            se_ghi = (pred_ghi[b] - target_ghi[b]) ** 2                # [T,H,W]
            if mask is not None:
                m = mask[b]                                            # [T,H,W]
                se_sum_t = (se_ghi * m).sum(dim=(1, 2))                # [T]
                cnt_t = m.sum(dim=(1, 2)).clamp_min(1.0)               # [T]
                ghi_mse_t = se_sum_t / cnt_t
            else:
                ghi_mse_t = se_ghi.mean(dim=(1, 2))                    # [T]
            ghi_rmse_t = torch.sqrt(ghi_mse_t)                         # [T]

            # ---- accumulate (running mean via sum/count) ----
            self.event_cloud_rmse_sum[event_time] += cloud_rmse_t
            self.event_ghi_rmse_sum[event_time]   += ghi_rmse_t
            self.event_cnt[event_time] += 1

    def get_event_table(self, which="ghi") -> Dict[str, torch.Tensor]:
        """
        返回 event -> rmse_t[T] (float32)
        which: 'ghi' or 'cloud'
        """
        out = {}
        sums = self.event_ghi_rmse_sum if which == "ghi" else self.event_cloud_rmse_sum
        for event in sorted(self.event_cnt.keys()):
            out[event] = (sums[event] / max(self.event_cnt[event], 1)).to(torch.float32)
        return out

    def compute(self, mode="leadtime", prefix="val") -> Dict[str, float]:
        """
        汇总所有 event 的 RMSE，返回用于 wandb 日志的字典。
        mode:
          - 'leadtime': 返回每个 leadtime 的 cloud/ghi RMSE 以及 overall RMSE
        """
        if mode == "leadtime":
            cloud_table = self.get_event_table(which="cloud")
            ghi_table = self.get_event_table(which="ghi")

            if len(cloud_table) == 0:
                return {f"{prefix}/cloud_rmse_overall": float('nan'),
                        f"{prefix}/ghi_rmse_overall": float('nan')}

            cloud_all = torch.stack(list(cloud_table.values()), dim=0)  # [N_events, T]
            ghi_all = torch.stack(list(ghi_table.values()), dim=0)      # [N_events, T]

            cloud_rmse_per_t = cloud_all.mean(dim=0)  # [T]
            ghi_rmse_per_t = ghi_all.mean(dim=0)      # [T]

            metrics = {
                f"{prefix}/cloud_rmse_overall": cloud_rmse_per_t.mean().item(),
                f"{prefix}/ghi_rmse_overall": ghi_rmse_per_t.mean().item(),
            }
            for t in range(self.num_frames):
                metrics[f"{prefix}/cloud_rmse_t{t+1:02d}"] = cloud_rmse_per_t[t].item()
                metrics[f"{prefix}/ghi_rmse_t{t+1:02d}"] = ghi_rmse_per_t[t].item()
            return metrics
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def export_event_table_csv(self, csv_path: str, which: str = "ghi"):
        tab = self.get_event_table(which=which)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = ["event_time\\leadtime"] + [str(i) for i in range(1, self.num_frames + 1)]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for event, rmse_t in tab.items():
                w.writerow([event] + [float(x) for x in rmse_t.cpu().tolist()])
