import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import shutil
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BaguanSolar import BaguanSolar
from datasets.cal_clear_ghi import compute_clearsky_ineichen_np
from datasets.two_stage_dataset import TwoStageDataset
from utils.plt import save_visualization_samples
from utils.lr_schedule import WarmupCosineScheduler
from utils.metrics import RMSEMetrics

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


EXPERIMENT_ROOT = "./experiments"

params_cfg = {
    "SateEncoderSwinNet":{
        "img_size": [512, 512],
        "patch_size": 8,
        "encoder": True,
        "decoder": False,
        "in_chans": 6*4,
        "out_chans": 256,
        "embed_dim": 256,
        "depths": [8],
        "num_heads": [2],
        "window_size": 16,
        "mlp_ratio": 4.,
        "qkv_bias": True,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "pretrained_window_sizes": [0, 0, 0, 0],
    },
    "EnvEncoderSwinNet":{
        "img_size": [512, 512],
        "patch_size": 8,
        "encoder": True,
        "decoder": False,
        "in_chans": 30*39,
        "out_chans": 256,
        "embed_dim": 256,
        "depths": [8],
        "num_heads": [2],
        "window_size": 16,
        "mlp_ratio": 4.,
        "qkv_bias": True,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "pretrained_window_sizes": [0, 0, 0, 0],
    },
    "SateCrossAttnBlock":{
        "embed_dim": 256,
        "num_heads": 8,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.1,
    },
    "MultiDecoderSwinNet_Stage1":{
        "img_size": [512, 512],
        "patch_size": 8,
        "encoder": False,
        "decoder": True,
        "in_chans": 512,
        "out_chans": 24*(4+1),
        "embed_dim": 512,
        "depths": [2],
        "num_heads": [2],
        "window_size": 16,
        "mlp_ratio": 4.,
        "qkv_bias": True,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "pretrained_window_sizes": [0, 0, 0, 0],
    },
    "MultiDecoderSwinNet_Stage2":{
        "img_size": [512, 512],
        "patch_size": 8,
        "encoder": True,
        "decoder": True,
        "in_chans": 3+4+11,
        "out_chans": 1,
        "embed_dim": 256,
        "depths": [8],
        "num_heads": [2],
        "window_size": 16,
        "mlp_ratio": 4.,
        "qkv_bias": True,
        "drop_rate": 0.,
        "attn_drop_rate": 0.,
        "drop_path_rate": 0.1,
        "norm_layer": nn.LayerNorm,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "pretrained_window_sizes": [0, 0, 0, 0],
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train OneTransformerNet for Solar GHI Prediction")
    # Training config
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_per_n_epoch", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--his_frames", type=int, default=6)
    parser.add_argument("--fut_frames", type=int, default=24)

    # Model config
    parser.add_argument("--aux_loss_weight", type=float, default=1.0)
    parser.add_argument("--target_size", type=int, default=512)

    parser.add_argument("--data_dir", type=str, default='./datasets/data/test')
    parser.add_argument("--era5_dir", type=str, default="./datasets/data/era5_test")
    parser.add_argument("--baguan_dir", type=str, default='./datasets/data/baguan_test_2025')
    parser.add_argument("--stats_path", type=str, default="./datasets/data_train_statistics.json")
    parser.add_argument("--era5_stats_path", type=str, default="./datasets/modify_era5_train.json")
    parser.add_argument("--latlon_path", type=str, default="./datasets/latlon_512x512.npy")

    parser.add_argument("--exp_dir", type=str, default='./experiments/BaguanSolar_V2')
    parser.add_argument("--load", type=str, default='best.ckpt')

    parser.add_argument("--exp_name", type=str, default="BaguanSolar_V2")
    parser.add_argument("--gpus", type=str, default="4", help="e.g., '0,1' or '2' (use first 2 GPUs)")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=16)

    return parser.parse_args()


def parse_gpus(gpus_arg):
    """
    Parse --gpus argument.
    - If gpus_arg is None -> return []
    - If gpus_arg is "2" -> [0, 1]
    - If gpus_arg is "0,1,3" -> [0, 1, 3]
    """
    if gpus_arg is None:
        return []
    try:
        if ',' in gpus_arg:
            device_ids = [int(x.strip()) for x in gpus_arg.split(',')]
        else:
            n = int(gpus_arg)
            device_ids = list(range(n))
        return device_ids
    except Exception as e:
        raise ValueError(f"Invalid --gpus format: {gpus_arg}. Use e.g., '0,1' or '2'. Error: {e}")


@torch.no_grad()
def validate(model, val_loader, aux_loss_weight, device, experiment_dir):
    model.eval()
    num_batches = len(val_loader)
    metrics = RMSEMetrics(num_frames=24, device=device)
    # samples_for_vis = []

    pbar = tqdm(val_loader, desc=f"Testing", leave=False, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        input_sat = batch["his_satellite"].permute(0, 2, 1, 3, 4).to(device)       # Stage1 卫星input  [-1, 1]
        input_clearghi = batch["fut_clearghi"].squeeze(2).to(device)   # Stage2 晴空GHI    [0, 1000]
        target_cloud_norm = batch["fut_cloud_label"].squeeze(2).to(device)  # Stage1 云量label  [-1, 1]
        target_ghi_norm = batch["fut_ghi_label"].squeeze(2).to(device)  # Stage2 观测GHI    [-1, 1]
        
        fut_ratio = batch["fut_ratio_label"].squeeze(2).to(device)
        event_name = batch["event_name"]

        his_era5 = batch["his_era5"].permute(0,2,1,3,4).contiguous().to(device) 
        # fut_era5 = batch["fut_era5"].permute(0,2,1,3,4).contiguous().to(device) 
        fut_baguan = batch["fut_baguan"].permute(0,2,1,3,4).contiguous().to(device) 
        times = batch["times"]

        total_era5 = torch.cat([his_era5, fut_baguan], dim=2)

        pred = model(input_sat, total_era5, input_clearghi) 

        # 逆归一化
        target_cloud_phys = (target_cloud_norm + 1) * 50
        target_ghi_phys   = (target_ghi_norm + 1) * 750
        pred_cloud_phys = pred["pred_cloud"]
        pred_ghi_phys   = pred["pred_ghi"]

        # Update metrics (in normalized space!)
        metrics.update(
            pred_cloud=pred_cloud_phys,
            target_cloud=target_cloud_phys,
            pred_ghi=pred_ghi_phys,
            target_ghi=target_ghi_phys,
            mask=None,
            time_list=times,
            loss_value=None,
        )

    metrics.export_event_table_csv(
        os.path.join(experiment_dir, "val_event_rmse_ghi.csv"),
        which="ghi"
    )
    metrics.export_event_table_csv(
        os.path.join(experiment_dir, "val_event_rmse_cloud.csv"),
        which="cloud"
    )



def test(args):
    # === GPU Setup ===
    device_ids = parse_gpus(args.gpus)
    if torch.cuda.is_available():
        if len(device_ids) == 0:
            # No --gpus specified: use single GPU or CPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Using single GPU: cuda:0" if torch.cuda.is_available() else "Using CPU")
        else:
            # Use specified GPUs
            for gid in device_ids:
                assert gid < torch.cuda.device_count(), f"GPU {gid} not available (total: {torch.cuda.device_count()})"
            device = torch.device(f"cuda:{device_ids[0]}")
            print(f"Using GPUs: {device_ids}")
    else:
        device = torch.device("cpu")
        device_ids = []
        print("Using CPU")

    # Load stats
    with open(args.stats_path) as f:
        stats = json.load(f)
    with open(args.era5_stats_path) as f:
        era5_stats = json.load(f)

    # Datasets
    test_ds = TwoStageDataset(
            data_dir=args.data_dir,
            era5_dir=args.era5_dir,
            baguan_dir=args.baguan_dir,
            latlon_path=args.latlon_path,
            stats=stats,
            era5_stats=era5_stats,
            history_frames=args.his_frames,
            future_frames=args.fut_frames, 
            split='test', 
            target_size=args.target_size,
        )
        
    print(f"The test set consists of {len(test_ds)} samples.")

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # geo_info = [train_ds.lat_arr, train_ds.lon_arr, train_ds.altitude]

    # Model
    model = BaguanSolar(
        params_cfg=params_cfg,
        target_size=512,
    ).to(device)

    experiment_dir = args.exp_dir
    os.makedirs(experiment_dir, exist_ok=True)

    ckpt_path = os.path.join(experiment_dir, "checkpoints", args.load)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    model.load_state_dict(new_state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=============================================")
    print(f"   Model Parameter Count:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable:        {total_params - trainable_params:,}")
    print(f"   Model Size (MB):      {total_params * 4 / (1024**2):.2f} MB")
    print(f"=============================================\n")

    # Multi-GPU wrap
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped with DataParallel on {len(device_ids)} GPUs")

    validate(
        model, test_loader, args.aux_loss_weight, device, experiment_dir
    )

if __name__ == "__main__":
    args = parse_args()
    test(args)
