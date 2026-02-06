# show_layer_importance.py
# 功能：计算 YOLO 各层的多任务梯度模长，并保存为 CSV 和可视化图表
# 运行示例：python show_layer_importance.py --weights best.pt --data straw.yaml --num_batches 20

import argparse
import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ================= 辅助函数区域 =================

def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    return x


def _load_yaml_dict(p: Path) -> dict:
    if not p.exists():
        raise RuntimeError(f"YAML file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="ignore")
    import yaml
    return yaml.safe_load(text)


def _merge_into_args(args_obj, d: dict):
    for k, v in d.items():
        if isinstance(k, str) and k.isidentifier() and not hasattr(args_obj, k):
            setattr(args_obj, k, v)


def _ensure_model_args_for_loss(model, hyp_yaml: str, weights_path: str):
    """
    全自动修复参数缺失问题，优先读取默认配置
    """
    # 1. 初始化基础配置 (从 default.yaml)
    base_args = {}
    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        base_args = vars(get_cfg(DEFAULT_CFG))
    except Exception as e:
        print(f"[Warning] Fallback to hardcoded defaults: {e}")
        base_args = {
            "task": "detect", "mode": "train", "imgsz": 640, "rect": False,
            "single_cls": False, "overlap_mask": True, "box": 7.5,
            "cls": 0.5, "dfl": 1.5, "pose": 12.0, "kobj": 1.0,
            "half": False, "cache": False, "augment": False, "fraction": 1.0
        }

    # 2. 加载 checkpoint 中的 args (优先级高于 default)
    w = Path(weights_path)
    cand_files = [w.parent / "args.yaml", w.parent.parent / "args.yaml"]
    for p in cand_files:
        if p.exists():
            try:
                d = _load_yaml_dict(p)
                d = {k: v for k, v in d.items() if v is not None}
                base_args.update(d)
                print(f"[Info] Loaded training args from: {p}")
            except:
                pass

    # 3. 赋值回 model.args
    if isinstance(getattr(model, "args", None), dict):
        model.args = _to_ns(model.args)
    if getattr(model, "args", None) is None:
        model.args = SimpleNamespace()

    for k, v in base_args.items():
        if not hasattr(model.args, k):
            setattr(model.args, k, v)

    # 4. 双重保险
    if not hasattr(model.args, "rect"): model.args.rect = False
    if not hasattr(model.args, "overlap_mask"): model.args.overlap_mask = True
    # 强制指定为 segpose 任务，确保加载 mask 和 keypoints
    model.args.task = "segpose"


def _move_batch_to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    img = out["img"]
    img = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
    out["img"] = img
    return out


def _build_block_groups(model: nn.Module):
    """按 model.0, model.1 等分组参数"""
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = {}
    for i, (name, _) in enumerate(named_params):
        seg = name.split(".")
        # 提取 model.x 作为组名
        if len(seg) >= 2 and seg[0] == "model" and seg[1].isdigit():
            gname = f"model.{seg[1]}"
        else:
            gname = "others"

        if gname not in groups:
            groups[gname] = []
        groups[gname].append(i)

    # 排序 keys
    def sort_key(k):
        if k.startswith("model."):
            try:
                return int(k.split(".")[1])
            except:
                return 9999
        return 9999

    sorted_keys = sorted(groups.keys(), key=sort_key)
    # 过滤掉 others 如果不想看
    sorted_keys = [k for k in sorted_keys if k != "others"]
    return named_params, {k: groups[k] for k in sorted_keys}


def _calc_group_norm(grads, idxs):
    """计算指定参数组的梯度 L2 Norm"""
    tensor_list = []
    for i in idxs:
        g = grads[i]
        if g is not None:
            tensor_list.append(g.flatten())

    if not tensor_list:
        return 0.0

    # 拼接后计算整体 norm
    cat = torch.cat(tensor_list)
    return torch.norm(cat).item()


class PatchedSegPoseLoss:
    def __init__(self, model):
        from ultralytics.utils.loss import v8SegmentationLoss, v8PoseLoss
        self.seg = v8SegmentationLoss(model)
        self.pose = v8PoseLoss(model)

    def __call__(self, preds, batch):
        if isinstance(preds, (list, tuple)) and len(preds) == 4:
            feats, mc, proto, kpt = preds
        elif isinstance(preds, (list, tuple)) and len(preds) == 2:
            feats, mc, proto, kpt = preds[1]
        else:
            raise ValueError("Unknown preds format")

        seg_loss_bs, _ = self.seg((feats, mc, proto), batch)
        pose_loss_bs, _ = self.pose((feats, kpt), batch)

        # loss_vec: [box, seg, pose, kobj, cls, dfl]
        loss_vec = torch.stack([
            seg_loss_bs[0],  # box
            seg_loss_bs[1],  # seg
            pose_loss_bs[1],  # pose
            pose_loss_bs[2],  # kobj
            seg_loss_bs[2],  # cls
            seg_loss_bs[3]  # dfl
        ])
        return loss_vec


# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='runs/segment/csp/weights/best.pt')
    parser.add_argument("--data", type=str, default='ultralytics/cfg/datasets/straw.yaml')
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--save_dir", type=str, default="runs/segment/csp/layer_importance")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
        from ultralytics.data.build import build_yolo_dataset, build_dataloader
        from ultralytics.data.utils import check_det_dataset
    except ImportError:
        raise RuntimeError("Please run in ultralytics environment")

    # Load Model
    y = YOLO(args.weights, task="segpose")
    model = y.model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 强制解冻并移除 hook
    for p in model.parameters():
        p.requires_grad = True
    if hasattr(model, "grad_balancer"): setattr(model, "grad_balancer", None)

    # 修复参数 + 强制设置任务类型
    _ensure_model_args_for_loss(model, "", args.weights)

    criterion = PatchedSegPoseLoss(model)
    model.train()

    # Dataloader
    data_cfg = check_det_dataset(args.data)
    dataset = build_yolo_dataset(
        cfg=model.args, img_path=data_cfg[args.split], batch=args.batch,
        data=data_cfg, mode=args.split, rect=False, stride=32
    )
    loader = build_dataloader(dataset, batch=args.batch, workers=0, shuffle=False)

    # Groups
    named_params, groups = _build_block_groups(model)
    all_params = [p for _, p in named_params]

    # Stats Containers
    raw_stats = {g: {'det': 0.0, 'seg': 0.0, 'pose': 0.0} for g in groups}
    counts = 0

    print(f"Start analyzing {args.num_batches} batches...")

    for i, batch in enumerate(loader):
        if i >= args.num_batches: break

        batch = _move_batch_to_device(batch, device)
        preds = model(batch["img"])
        loss_vec = criterion(preds, batch)

        # Loss Decomposition
        # Det = box + cls + dfl
        loss_det = loss_vec[0] + loss_vec[4] + loss_vec[5]
        # Seg = seg
        loss_seg = loss_vec[1]
        # Pose = pose + kobj
        loss_pose = loss_vec[2] + loss_vec[3]

        # Backprop
        g_det = torch.autograd.grad(loss_det, all_params, retain_graph=True, allow_unused=True)
        g_seg = torch.autograd.grad(loss_seg, all_params, retain_graph=True, allow_unused=True)
        g_pose = torch.autograd.grad(loss_pose, all_params, allow_unused=True)

        # Calculate Norms per Group
        for gname, idxs in groups.items():
            raw_stats[gname]['det'] += _calc_group_norm(g_det, idxs)
            raw_stats[gname]['seg'] += _calc_group_norm(g_seg, idxs)
            raw_stats[gname]['pose'] += _calc_group_norm(g_pose, idxs)

        counts += 1
        print(f"Processed batch {i + 1}/{args.num_batches}", end="\r")

    # Average
    labels = list(groups.keys())
    det_vals = [raw_stats[l]['det'] / counts for l in labels]
    seg_vals = [raw_stats[l]['seg'] / counts for l in labels]
    pose_vals = [raw_stats[l]['pose'] / counts for l in labels]

    # ---- 1. Save Detailed CSV (新增部分) ----
    csv_path = save_dir / "layer_importance_detailed.csv"
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header: Layer, Raw Norms, Relative Ratios
        writer.writerow([
            "Layer",
            "Det_Norm", "Seg_Norm", "Pose_Norm", "Total_Norm",
            "Det_Ratio(%)", "Seg_Ratio(%)", "Pose_Ratio(%)"
        ])

        for idx, layer_name in enumerate(labels):
            d = det_vals[idx]
            s = seg_vals[idx]
            p = pose_vals[idx]
            total = d + s + p + 1e-8  # 防止除零

            writer.writerow([
                layer_name,
                f"{d:.4f}", f"{s:.4f}", f"{p:.4f}", f"{total:.4f}",
                f"{d / total * 100:.2f}", f"{s / total * 100:.2f}", f"{p / total * 100:.2f}"
            ])

    print(f"\n[OK] CSV saved to {csv_path}")

    # ---- 2. Plotting ----
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(14, 7))  # 稍微加宽一点
    plt.bar(x - width, det_vals, width, label='Det', color='#1f77b4', alpha=0.9)
    plt.bar(x, seg_vals, width, label='Seg', color='#ff7f0e', alpha=0.9)
    plt.bar(x + width, pose_vals, width, label='Pose', color='#2ca02c', alpha=0.9)

    plt.ylabel('Avg Gradient Norm (L2)')
    plt.title('Layer-wise Task Importance (Gradient Magnitude)')
    plt.xticks(x, labels, rotation=90)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    out_png = save_dir / "layer_importance_norm.png"
    plt.savefig(out_png, dpi=300)
    print(f"[OK] Plot saved to {out_png}")


if __name__ == "__main__":
    main()