# grad_conflict.py
# 目标：加载 baseline 权重，在 val/train 上抽 N 个 batch，按 det/seg/pose 计算各层（按 model.<idx> 分组）梯度冲突，
# 输出：CSV + PNG（overall_conflict_mean 柱状图，越大越冲突）
#
# 运行示例：
# python grad_conflict.py --weights ../runs/segment/straw/weights/best.pt --data ultralytics/cfg/datasets/straw.yaml --split val --num_batches 50 --batch 4 --save_dir runs/grad_conflict/baseline
#
# 可选：如果 checkpoint 没保存完整 args，可用 --hyp_yaml 指定训练时的 args.yaml/hyp.yaml 用于补齐 loss 所需关键超参

import os
import csv
import math
import time
import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    try:
        import yaml  # type: ignore
        d = yaml.safe_load(text)
    except Exception:
        try:
            from ruamel.yaml import YAML  # type: ignore
            y = YAML(typ="safe")
            d = y.load(text)
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML: {p}. Error: {e}")

    if not isinstance(d, dict):
        raise RuntimeError(f"YAML file is not a dict: {p}")
    return d


def _merge_into_args(args_obj, d: dict, overwrite: bool = False):
    for k, v in d.items():
        if not isinstance(k, str) or (not k.isidentifier()):
            continue
        if overwrite or (not hasattr(args_obj, k)):
            setattr(args_obj, k, v)


def _ensure_model_args_for_loss(model, hyp_yaml: str, weights_path: str):
    # 修复 model.args 为 dict 导致的点访问失败
    if isinstance(getattr(model, "args", None), dict):
        model.args = _to_ns(model.args)
    if getattr(model, "args", None) is None:
        model.args = SimpleNamespace()

    # v8SegmentationLoss/v8PoseLoss 常用关键字段（你的 loss.py 会用）
    need = ["box", "cls", "dfl", "pose", "kobj", "overlap_mask"]
    missing = [k for k in need if not hasattr(model.args, k)]
    if not missing:
        return

    cand_files = []
    if hyp_yaml:
        cand_files.append(Path(hyp_yaml))

    w = Path(weights_path)
    cand_files.append(w.parent.parent / "args.yaml")
    cand_files.append(w.parent.parent / "hyp.yaml")

    loaded = False
    for p in cand_files:
        if p.exists():
            d = _load_yaml_dict(p)
            _merge_into_args(model.args, d, overwrite=False)
            loaded = True

    missing2 = [k for k in need if not hasattr(model.args, k)]
    if missing2:
        msg = (
            f"model.args missing keys required by loss: {missing2}. "
            f"Provide the exact training args via --hyp_yaml (args.yaml/hyp.yaml), "
            f"or ensure they are saved in checkpoint/model.args."
        )
        if loaded:
            msg += " (Tried loading args.yaml/hyp.yaml but still missing.)"
        raise RuntimeError(msg)


def _call_with_supported_kwargs(fn, kwargs: dict):
    # 兼容不同 fork 的函数签名：只把支持的 kwargs 传进去
    sig = inspect.signature(fn)
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return fn(**filtered)


def _move_batch_to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    if "img" not in out:
        raise KeyError("batch missing key 'img'. Your dataloader must return a dict with 'img'.")

    img = out["img"]
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    elif img.dtype in (torch.float16, torch.float32, torch.float64):
        if img.numel() > 0 and float(img.max().item()) > 1.5:
            img = img / 255.0
        img = img.float()
    else:
        img = img.float()

    out["img"] = img
    return out


def _freeze_bn(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def _build_block_groups(model: nn.Module):
    """
    按 Ultralytics 常见命名：model.<idx>.* 分组。
    不符合则 fallback 为 name 的第一个段。
    """
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = {}
    for i, (name, _) in enumerate(named_params):
        seg = name.split(".")
        if len(seg) >= 3 and seg[0] == "model" and seg[1].isdigit():
            gname = f"model.{seg[1]}"
        else:
            gname = seg[0]
        groups.setdefault(gname, []).append(i)
    return named_params, groups


def _pair_cos_and_stats(grads_a, grads_b, idxs, eps=1e-12):
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    used = 0

    for i in idxs:
        ga = grads_a[i]
        gb = grads_b[i]
        if ga is None or gb is None:
            continue
        ga = ga.float()
        gb = gb.float()
        dot += float((ga * gb).sum().item())
        na2 += float((ga * ga).sum().item())
        nb2 += float((gb * gb).sum().item())
        used += 1

    if used == 0 or na2 <= eps or nb2 <= eps:
        return None

    cos = dot / (math.sqrt(na2) * math.sqrt(nb2) + eps)
    conflict = max(0.0, -cos)  # 0 表示同向/不冲突，越接近 1 越接近反向
    return cos, conflict, used


def _resolve_data_dict(data_yaml_path: str):
    # 优先用 ultralytics 的 check_det_dataset；不兼容就自己读 yaml
    try:
        from ultralytics.data.utils import check_det_dataset
        data = check_det_dataset(data_yaml_path)
        if not isinstance(data, dict):
            raise RuntimeError("check_det_dataset did not return a dict.")
        return data
    except Exception:
        d = _load_yaml_dict(Path(data_yaml_path))
        base = d.get("path", "")
        if isinstance(base, str) and base:
            base = os.path.expanduser(base)
            base = os.path.abspath(base)
            for k in ["train", "val", "test"]:
                if k in d and isinstance(d[k], str) and not os.path.isabs(d[k]):
                    d[k] = os.path.join(base, d[k])
        return d


def main():
    parser = argparse.ArgumentParser()

    # ✅你要求保留的两行默认值
    parser.add_argument("--weights", type=str, default="../runs/segment/straw/weights/best.pt")
    parser.add_argument("--data", type=str, default="ultralytics/cfg/datasets/straw.yaml")

    parser.add_argument("--hyp_yaml", type=str, default="")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="runs/grad_conflict")
    parser.add_argument("--plot_topk", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.weights:
        raise ValueError("Missing --weights.")
    if not args.data:
        raise ValueError("Missing --data.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_csv = save_dir / "grad_conflict_by_block.csv"
    out_png = save_dir / "grad_conflict_by_block.png"

    _set_seed(int(args.seed))

    # ---- imports from your repo env ----
    try:
        from ultralytics import YOLO
        try:
            from ultralytics.data.build import build_yolo_dataset, build_dataloader
        except Exception:
            from ultralytics.data import build_yolo_dataset, build_dataloader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Import ultralytics failed. Run this script inside your repo python env.\n"
            f"Original error: {e}"
        )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and str(args.device).lower() != "cpu" else "cpu"
    )

    # ---- load model ----
    y = YOLO(args.weights, task="segpose")
    model = y.model
    model.to(device)

    # 关闭任何 grad balance（测 baseline 原始梯度冲突）
    if hasattr(model, "grad_balancer"):
        setattr(model, "grad_balancer", None)
    if hasattr(model, "grad_balance_shared_params"):
        setattr(model, "grad_balance_shared_params", None)

    # 修复 model.args 并补齐 loss 关键超参（供 model(batch) 内部 loss 使用）
    _ensure_model_args_for_loss(model, args.hyp_yaml, args.weights)

    # 训练态 forward（关键：model(batch) 走训练同路径，不走推理 no_grad）
    model.train()
    _freeze_bn(model)

    # ---- build dataloader ----
    data = _resolve_data_dict(args.data)
    if args.split not in data:
        raise KeyError(f"Split '{args.split}' not found. Available keys: {list(data.keys())}")

    stride_val = 32
    if hasattr(model, "stride"):
        try:
            stride_val = int(getattr(model, "stride").max().item())
        except Exception:
            stride_val = 32

    cfg_obj = getattr(y, "args", None)
    if cfg_obj is None:
        cfg_obj = getattr(model, "args", None)

    dataset_kwargs = dict(
        cfg=cfg_obj,
        img_path=data[args.split],
        batch=int(args.batch),
        data=data,
        mode=args.split,
        rect=False,
        stride=stride_val,
        pad=0.0,                 # 你的 fork 不支持会被过滤
        prefix=f"{args.split}: ",
        imgsz=int(args.imgsz),    # 你的 fork 不支持会被过滤
    )
    dataset = _call_with_supported_kwargs(build_yolo_dataset, dataset_kwargs)

    loader_kwargs = dict(
        dataset=dataset,
        batch=int(args.batch),
        workers=int(args.workers),
        shuffle=False,
        rank=-1,
    )
    loader = _call_with_supported_kwargs(build_dataloader, loader_kwargs)

    named_params, groups = _build_block_groups(model)
    all_params = [p for _, p in named_params]

    # ---- accumulators ----
    stats = {}
    for g in groups.keys():
        stats[g] = {
            "n_batches": 0,
            "det_seg_cos_sum": 0.0,
            "det_pose_cos_sum": 0.0,
            "seg_pose_cos_sum": 0.0,
            "det_seg_conf_sum": 0.0,
            "det_pose_conf_sum": 0.0,
            "seg_pose_conf_sum": 0.0,
            "det_seg_neg": 0,
            "det_pose_neg": 0,
            "seg_pose_neg": 0,
            "det_seg_valid": 0,
            "det_pose_valid": 0,
            "seg_pose_valid": 0,
        }

    it = iter(loader)
    t0 = time.time()
    used_batches = 0

    torch.set_grad_enabled(True)
    if not torch.is_grad_enabled():
        raise RuntimeError("Grad is disabled globally (torch.set_grad_enabled(False) somewhere).")

    for _ in range(int(args.num_batches)):
        try:
            batch = next(it)
        except StopIteration:
            break

        batch = _move_batch_to_device(batch, device)

        # ✅关键改动：直接走训练同路径，拿到需要反传的 6 维 loss 向量
        loss_vec, loss_items = model(batch)

        if (not torch.is_tensor(loss_vec)) or loss_vec.numel() != 6:
            raise RuntimeError(
                f"model(batch) must return a 6-dim loss vector for segpose, got: {type(loss_vec)} shape={getattr(loss_vec, 'shape', None)}"
            )
        if not loss_vec.requires_grad:
            raise RuntimeError(
                "loss_vec does not require grad. That means your model(batch) path is still under no_grad/inference_mode.\n"
                "But trainer.py uses model(batch) for backward, so this indicates your environment/script disabled grad globally."
            )

        # segpose3 decomposition（与你 loss.py 的 segpose3 定义一致）
        loss_det = loss_vec[0] + loss_vec[4] + loss_vec[5]      # box + cls + dfl
        loss_seg = loss_vec[1]                                  # seg
        loss_pose = loss_vec[2] + loss_vec[3]                   # pose + kobj

        grads_det = torch.autograd.grad(loss_det, all_params, retain_graph=True, create_graph=False, allow_unused=True)
        grads_seg = torch.autograd.grad(loss_seg, all_params, retain_graph=True, create_graph=False, allow_unused=True)
        grads_pose = torch.autograd.grad(loss_pose, all_params, retain_graph=False, create_graph=False, allow_unused=True)

        for gname, idxs in groups.items():
            r = _pair_cos_and_stats(grads_det, grads_seg, idxs)
            if r is not None:
                cos, conf, _ = r
                stats[gname]["det_seg_cos_sum"] += cos
                stats[gname]["det_seg_conf_sum"] += conf
                stats[gname]["det_seg_valid"] += 1
                if cos < 0:
                    stats[gname]["det_seg_neg"] += 1

            r = _pair_cos_and_stats(grads_det, grads_pose, idxs)
            if r is not None:
                cos, conf, _ = r
                stats[gname]["det_pose_cos_sum"] += cos
                stats[gname]["det_pose_conf_sum"] += conf
                stats[gname]["det_pose_valid"] += 1
                if cos < 0:
                    stats[gname]["det_pose_neg"] += 1

            r = _pair_cos_and_stats(grads_seg, grads_pose, idxs)
            if r is not None:
                cos, conf, _ = r
                stats[gname]["seg_pose_cos_sum"] += cos
                stats[gname]["seg_pose_conf_sum"] += conf
                stats[gname]["seg_pose_valid"] += 1
                if cos < 0:
                    stats[gname]["seg_pose_neg"] += 1

            stats[gname]["n_batches"] += 1

        used_batches += 1

    if used_batches == 0:
        raise RuntimeError("No batches were processed. Check your --data yaml and split path.")

    elapsed = time.time() - t0

    def _mean(sumv, cnt):
        return sumv / cnt if cnt > 0 else float("nan")

    rows = []
    for gname, s in stats.items():
        det_seg_cos = _mean(s["det_seg_cos_sum"], s["det_seg_valid"])
        det_pose_cos = _mean(s["det_pose_cos_sum"], s["det_pose_valid"])
        seg_pose_cos = _mean(s["seg_pose_cos_sum"], s["seg_pose_valid"])

        det_seg_conf = _mean(s["det_seg_conf_sum"], s["det_seg_valid"])
        det_pose_conf = _mean(s["det_pose_conf_sum"], s["det_pose_valid"])
        seg_pose_conf = _mean(s["seg_pose_conf_sum"], s["seg_pose_valid"])

        conf_list = [v for v in [det_seg_conf, det_pose_conf, seg_pose_conf] if not math.isnan(v)]
        overall_conf = sum(conf_list) / len(conf_list) if conf_list else float("nan")

        det_seg_neg_frac = (s["det_seg_neg"] / s["det_seg_valid"]) if s["det_seg_valid"] > 0 else float("nan")
        det_pose_neg_frac = (s["det_pose_neg"] / s["det_pose_valid"]) if s["det_pose_valid"] > 0 else float("nan")
        seg_pose_neg_frac = (s["seg_pose_neg"] / s["seg_pose_valid"]) if s["seg_pose_valid"] > 0 else float("nan")

        rows.append({
            "group": gname,
            "n_batches": s["n_batches"],
            "overall_conflict_mean": overall_conf,
            "det_seg_cos_mean": det_seg_cos,
            "det_pose_cos_mean": det_pose_cos,
            "seg_pose_cos_mean": seg_pose_cos,
            "det_seg_conflict_mean": det_seg_conf,
            "det_pose_conflict_mean": det_pose_conf,
            "seg_pose_conflict_mean": seg_pose_conf,
            "det_seg_neg_frac": det_seg_neg_frac,
            "det_pose_neg_frac": det_pose_neg_frac,
            "seg_pose_neg_frac": seg_pose_neg_frac,
            "det_seg_valid": s["det_seg_valid"],
            "det_pose_valid": s["det_pose_valid"],
            "seg_pose_valid": s["seg_pose_valid"],
        })

    rows_sorted = sorted(
        rows,
        key=lambda x: (-(x["overall_conflict_mean"] if not math.isnan(x["overall_conflict_mean"]) else -1e9))
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow(r)

    topk = max(1, int(args.plot_topk))
    plot_rows = rows_sorted[:topk]
    xs = [r["group"] for r in plot_rows]
    ys = [r["overall_conflict_mean"] for r in plot_rows]

    plt.figure(figsize=(max(10, int(0.35 * len(xs))), 5))
    plt.bar(xs, ys)
    plt.xticks(rotation=90)
    plt.ylabel("overall_conflict_mean = mean(max(0, -cos)) over task pairs")
    plt.title(f"Gradient Conflict by Block (batches={used_batches}, time={elapsed:.1f}s)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] processed_batches={used_batches} time={elapsed:.1f}s device={device}")
    print(f"[OK] csv: {out_csv}")
    print(f"[OK] png: {out_png}")


if __name__ == "__main__":
    main()
