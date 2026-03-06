# torchrun --nproc_per_node=8 train_BaguanSolar.py
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import wandb
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
from einops import rearrange
import schedulefree

from model.BaguanSolar import BaguanSolar
from datasets.cal_clear_ghi import compute_clearsky_ineichen_np
from datasets.two_stage_dataset import TwoStageDataset
from utils.plt import save_visualization_samples
from utils.lr_schedule import WarmupCosineScheduler
from utils.metrics import RMSEMetrics
from utils.pt_load import load_state_dict_flexible

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def ddp_mean_dict(metrics_dict, device):
    """
    对 metrics_dict 里所有 (int/float) 数值做 all_reduce mean。
    非数值项会原样返回（一般你这里没有）。
    """
    if (not dist.is_available()) or (not dist.is_initialized()):
        return metrics_dict

    world_size = dist.get_world_size()
    out = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float, np.floating)):
            t = torch.tensor(float(v), device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = (t / world_size).item()
        else:
            out[k] = v
    return out


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="wandb")


# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_per_n_epoch", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--his_frames", type=int, default=6)
    parser.add_argument("--fut_frames", type=int, default=24)
    # REMOVED: --max_train_steps

    # Model config
    parser.add_argument("--aux_loss_weight", type=float, default=0.1)
    parser.add_argument("--target_size", type=int, default=512)

    # System
    parser.add_argument("--data_dir", type=str, default="/data/na130/SSRA_PREDICTION/Data/train")
    parser.add_argument("--era5_dir", type=str, default="/data/na130/SSRA_PREDICTION/Data/train_era5")
    parser.add_argument("--stats_path", type=str, default="./datasets/data_train_statistics.json")
    parser.add_argument("--era5_stats_path", type=str, default="./datasets/modify_era5_train.json")
    parser.add_argument("--latlon_path", type=str, default="/data/na130/SSRA_PREDICTION/latlon_512x512.npy")

    parser.add_argument("--exp_name", type=str, default="BaguanSolar_V2")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="SSRA_CS_GHI_6h_v2")
    parser.add_argument("--wandb_entity", type=str, default="maziqing_team")
    parser.add_argument("--wandb_group", type=str, default="solarseer")
    parser.add_argument("--wandb_name", type=str, default="[BaguanSolar]BaguanSolar_V2")

    return parser.parse_args()


def train_one_step(model, batch, optimizer, scaler, use_amp, aux_loss_weight, device):
    input_sat = batch["his_satellite"].permute(0,2,1,3,4).contiguous().to(device)       # Stage1 卫星input  [-1, 1]
    target_sat = batch["fut_satellite"].contiguous().to(device)       # Stage1 卫星input  [-1, 1]
    input_clearghi = batch["fut_clearghi"].squeeze(2).to(device)   # Stage2 晴空GHI    [0, 1]
    target_cloud = batch["fut_cloud_label"].squeeze(2).to(device)  # Stage1 云量label  [-1, 1]
    target_obs_ghi = batch["fut_ghi_label"].squeeze(2).to(device)  # Stage2 观测GHI    [-1, 1]
    fut_ratio = batch["fut_ratio_label"].squeeze(2).to(device)
    
    his_era5 = batch["his_era5"].permute(0,2,1,3,4).contiguous().to(device) 
    fut_era5 = batch["fut_era5"].permute(0,2,1,3,4).contiguous().to(device) 
    times = batch["times"]

    total_era5 = torch.cat([his_era5, fut_era5], dim=2)
    optimizer.zero_grad()

    if use_amp and scaler is not None:
        with torch.cuda.amp.autocast():
            pred = model(input_sat, total_era5, input_clearghi)

            mask = (torch.abs(fut_ratio) >= 0.0001).float()
            wmask = (input_clearghi / (input_clearghi.mean(dim=(-1, -2, -3, -4), keepdim=True) + 1e-8)).clip(0, 10)

            loss_satellite = F.mse_loss(pred["pred_satellite"], target_sat)
            loss_cloud = F.mse_loss(pred["pred_cloud"] / 50. - 1, target_cloud)
            loss_ghi = (F.mse_loss(pred["pred_ghi"] / 750. - 1, target_obs_ghi, reduction='none') * mask * wmask).mean()
            loss = aux_loss_weight * loss_cloud + loss_satellite + loss_ghi 

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        pred = model(input_sat, total_era5, input_clearghi)

        mask = (torch.abs(fut_ratio) >= 0.0001).float()
        wmask = (input_clearghi / (input_clearghi.mean(dim=(-1, -2, -3, -4), keepdim=True) + 1e-8)).clip(0, 10)

        loss_satellite = F.mse_loss(pred["pred_satellite"], target_sat)
        loss_cloud = F.mse_loss(pred["pred_cloud"] / 50. - 1, target_cloud)
        loss_ghi = (F.mse_loss(pred["pred_ghi"] / 750. - 1, target_obs_ghi, reduction='none') * mask * wmask).mean()
        loss = aux_loss_weight * loss_cloud + loss_satellite + loss_ghi 

        loss.backward()
        optimizer.step()

    target_cloud = (target_cloud + 1) * 50.     # [0, 100]
    target_obs_ghi   = (target_obs_ghi + 1) * 750.    # [0, 1500]

    return {
        'total_loss': loss.item(),
        'cloud_loss': loss_cloud,
        'ghi_loss': loss_ghi.item(),
        'satellite_loss': loss_satellite
    }


@torch.no_grad()
def validate(model, val_loader, aux_loss_weight, device, epoch, experiment_dir):
    model.eval()
    metrics = RMSEMetrics(num_frames=24, device=device)

    samples_for_vis = [] if is_main_process() else None

    pbar = tqdm(
        val_loader,
        desc=f"Validating Epoch {epoch+1}",
        leave=False,
        dynamic_ncols=True,
        disable=not is_main_process()
    )

    for idx, batch in enumerate(pbar):
        input_sat = batch["his_satellite"].permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
        target_sat = batch["fut_satellite"].contiguous().to(device, non_blocking=True)

        input_clearghi = batch["fut_clearghi"].squeeze(2).contiguous().to(device, non_blocking=True)
        target_cloud = batch["fut_cloud_label"].squeeze(2).contiguous().to(device, non_blocking=True)
        target_obs_ghi = batch["fut_ghi_label"].squeeze(2).contiguous().to(device, non_blocking=True)
        fut_ratio = batch["fut_ratio_label"].squeeze(2).contiguous().to(device, non_blocking=True)

        his_era5 = batch["his_era5"].permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
        fut_era5 = batch["fut_era5"].permute(0, 2, 1, 3, 4).contiguous().to(device, non_blocking=True)
        times = batch["times"]

        total_era5 = torch.cat([his_era5, fut_era5], dim=2)

        pred = model(input_sat, total_era5, input_clearghi)

        mask = (torch.abs(fut_ratio) >= 1e-4).float()
        wmask = (input_clearghi / (input_clearghi.mean(dim=(-1, -2, -3, -4), keepdim=True) + 1e-8)).clamp(0, 10)

        loss_satellite = F.mse_loss(pred["pred_satellite"], target_sat)
        loss_cloud = F.mse_loss(pred["pred_cloud"] / 50. - 1, target_cloud)
        loss_ghi = (F.mse_loss(pred["pred_ghi"] / 750. - 1, target_obs_ghi, reduction='none') * mask * wmask).mean()
        total_loss = aux_loss_weight * (loss_cloud + loss_satellite) + loss_ghi

        target_cloud_denorm = (target_cloud + 1) * 50.
        target_ghi_denorm = (target_obs_ghi + 1) * 750.

        metrics.update(
            pred_cloud=pred["pred_cloud"],
            target_cloud=target_cloud_denorm,
            pred_ghi=pred["pred_ghi"],
            target_ghi=target_ghi_denorm,
            mask=None,
            time_list=times,
            loss_value=total_loss.item(),
        )

        if is_main_process() and idx < 5:
            event_name = batch["event_name"]
            samples_for_vis.append((
                input_sat[0].detach().cpu().numpy(),
                target_cloud_denorm[0].detach().cpu().numpy(),
                target_ghi_denorm[0].detach().cpu().numpy(),
                pred["pred_cloud"][0].detach().cpu().numpy(),
                pred["pred_ghi"][0].detach().cpu().numpy(),
                event_name[0],
            ))

    metrics_dict_local = metrics.compute(mode="leadtime")
    metrics_dict = ddp_mean_dict(metrics_dict_local, device=device)
    if is_main_process():
        examples_dir = os.path.join(experiment_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        save_visualization_samples(samples_for_vis, epoch, examples_dir)
        
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return metrics_dict



def pt_save(model, model_path, epoch, val_loss, best_val):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    ckpt_name = f"epoch_{epoch}_RMSE_{val_loss:.5f}.ckpt"
    ckpt_path = os.path.join(model_path, ckpt_name)
    torch.save(model_to_save.state_dict(), ckpt_path)

    # last.ckpt
    last_path = os.path.join(model_path, "last.ckpt")
    if os.path.exists(last_path):
        os.remove(last_path)
    try:
        os.symlink(ckpt_name, last_path)
    except (OSError, NotImplementedError):
        shutil.copy2(ckpt_path, last_path)

    # best.ckpt
    if val_loss < best_val:
        best_path = os.path.join(model_path, "best.ckpt")
        if os.path.exists(best_path):
            os.remove(best_path)
        try:
            os.symlink(ckpt_name, best_path)
        except (OSError, NotImplementedError):
            shutil.copy2(ckpt_path, best_path)

    return val_loss < best_val


def train(args):
    # === GPU Setup ===
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # Load stats
    with open(args.stats_path) as f:
        stats = json.load(f)
    with open(args.era5_stats_path) as f:
        era5_stats = json.load(f)

    # Datasets
    train_ds = TwoStageDataset(
            data_dir=args.data_dir,
            era5_dir=args.era5_dir,
            latlon_path=args.latlon_path,
            stats=stats,
            era5_stats=era5_stats,
            history_frames=args.his_frames,
            future_frames=args.fut_frames, 
            split='train', 
            target_size=args.target_size
        )
    val_ds = TwoStageDataset(
            data_dir=args.data_dir,
            era5_dir=args.era5_dir,
            latlon_path=args.latlon_path,
            stats=stats,
            era5_stats=era5_stats,
            history_frames=args.his_frames,
            future_frames=args.fut_frames, 
            split='val', 
            target_size=args.target_size
        )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)

    print(f"The training set consists of {len(train_ds)} samples.")
    print(f"The validation set consists of {len(val_ds)} samples.")

    train_loader = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, sampler=val_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = BaguanSolar(
        params_cfg=params_cfg,
        target_size=512,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=============================================")
    print(f"   Model Parameter Count:")
    print(f"   Total Parameters:     {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")
    print(f"   Non-trainable:        {total_params - trainable_params:,}")
    print(f"   Model Size (MB):      {total_params * 4 / (1024**2):.2f} MB")
    print(f"=============================================\n")

    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=args.lr,          # 比如 1e-3、3e-4 等
        weight_decay=1e-2,   # 与 AdamW 一样配置
    )
    total_steps = args.epochs * len(train_loader)
    # scheduler = WarmupCosineScheduler(
    #     optimizer=optimizer,
    #     warmup_epochs=0.02*total_steps,          # 前 2 个 epoch 线性 warmup
    #     total_epochs=total_steps, # 总训练 epoch 数
    #     min_lr=1e-6               # 最终降到 1e-6
    # )
    scheduler = None
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = os.path.join(EXPERIMENT_ROOT, f"{args.exp_name}_{timestamp}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存配置 args 到 cfg.yaml
    cfg_path = os.path.join(experiment_dir, "cfg.yaml")
    if is_main_process():
        with open(cfg_path, 'w') as f:
            # 将 argparse.Namespace 转为 dict 并保存
            yaml.dump(vars(args), f, default_flow_style=False, indent=4, sort_keys=False)
        
        # WandB
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{args.wandb_name}_{timestamp}",
            # id="gix4gpqu",
            # resume="allow",
            config=vars(args)
        )

        def make_loggable_cfg(cfg):
            cfg = cfg.copy()
            for k, v in cfg.items():
                if isinstance(v, type):  # 类，例如 nn.LayerNorm
                    cfg[k] = v.__name__
            return cfg

        log_params_cfg = {
            "SateEncoderSwinNet": make_loggable_cfg(params_cfg["SateEncoderSwinNet"]),
            "EnvEncoderSwinNet": make_loggable_cfg(params_cfg["EnvEncoderSwinNet"]),
            "SateCrossAttnBlock": make_loggable_cfg(params_cfg["SateCrossAttnBlock"]),
            "MultiDecoderSwinNet_Stage1": make_loggable_cfg(params_cfg["MultiDecoderSwinNet_Stage1"]),
            "MultiDecoderSwinNet_Stage2": make_loggable_cfg(params_cfg["MultiDecoderSwinNet_Stage2"]),
            "Model_Size_MB": int(total_params * 4 / (1024**2))
        }
        wandb.config.update(log_params_cfg)

    best_val = float('inf')
    for epoch in range(args.epochs):
        # --- Training ---
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.train()
        total_loss, cloud_loss, ghi_loss = 0.0, 0.0, 0.0
        num_train_steps = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            losses = train_one_step(model, batch, optimizer, scaler, args.use_amp, args.aux_loss_weight, device)
            total_loss += losses['total_loss']
            cloud_loss += losses['cloud_loss']
            ghi_loss += losses['ghi_loss']

            if scheduler is not None:
                scheduler.step()

            pbar.set_postfix({
                'loss': f"{losses['total_loss']:.3f}",
                'cloud_mse': f"{losses['cloud_loss']:.3f}",
                'ghi_mse': f"{losses['ghi_loss']:.3f}",
                'satellite_mse': f"{losses['satellite_loss']:.3f}"
            })
            if is_main_process():
                wandb.log({
                    "train/total_loss_batch": losses['total_loss'],
                    "train/cloud_loss_batch": losses['cloud_loss'],
                    "train/ghi_loss_batch": losses['ghi_loss'],
                    "train/satellite_loss_batch": losses['satellite_loss'],
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                }, step=epoch * num_train_steps + step)

                wandb.log({"train/epoch": epoch}, step=epoch * num_train_steps + step)
            
            break

        total_loss /= num_train_steps
        cloud_loss /= num_train_steps
        ghi_loss /= num_train_steps

        if is_main_process():
            wandb.log({
                "train/total_loss_epoch": total_loss,
                "train/cloud_loss_epoch": cloud_loss,
                "train/ghi_loss_epoch": ghi_loss
            }, step=(epoch + 1) * num_train_steps)

        optimizer.eval()
        # --- Validation ---
        do_val = ((epoch + 1) % args.val_per_n_epoch == 0) or (epoch == args.epochs - 1)
        if do_val:
            metrics_dict = validate(model, val_loader, args.aux_loss_weight, device, epoch, experiment_dir)
            if is_main_process():
                wandb.log(metrics_dict, step=(epoch + 1) * num_train_steps)
                is_best = pt_save(model, checkpoint_dir, epoch + 1, metrics_dict['val/ghi_rmse_overall'], best_val)
                if is_best:
                    best_val = metrics_dict['val/ghi_rmse_overall']
    if is_main_process():
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    train(args)
