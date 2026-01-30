from copy import deepcopy

import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.segpose.val import SegPoseValidator
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.gradbanlance import build_grad_balancer, get_shared_params
from ultralytics.utils.torch_utils import autocast, unwrap_model


class SegPoseTrainer(SegmentationTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segpose"
        # ✅让训练表头多出 pose_loss/kobj_loss
        self.loss_names = ("box_loss", "seg_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        self.conflict_metrics = {
            "conflict/seg_pose_cos": float("nan"),
            "conflict/seg_pose_angle": float("nan"),
            "conflict/seg_pose_neg": float("nan"),
            "conflict/det_seg_pose_cos": float("nan"),
            "conflict/det_seg_pose_angle": float("nan"),
            "conflict/det_seg_pose_neg": float("nan"),
        }
        self.add_callback("on_train_epoch_end", self._on_train_epoch_end)

    def set_model_attributes(self):
        super().set_model_attributes()
        # Expose sigma for keypoint loss to match OKS during validation.
        self.model.kpt_sigma = self.data.get("kpt_sigma")
        balancer = None
        try:
            balancer = build_grad_balancer(
                getattr(self.args, "grad_balance", None),
                alpha=getattr(self.args, "grad_balance_alpha", 1.5),
                lr=getattr(self.args, "grad_balance_lr", 0.025),
                interval=getattr(self.args, "grad_balance_interval", 1),
                warmup=getattr(self.args, "grad_balance_warmup", 0),
            )
        except Exception as exc:
            LOGGER.warning(f"grad_balance init failed, disabled: {exc}")

        self.model.grad_balancer = balancer
        if balancer is not None:
            self.model.grad_balance_groups = getattr(self.args, "grad_balance_groups", "segpose2")
            self.model.grad_balance_shared_params = get_shared_params(
                self.model, getattr(self.args, "grad_balance_shared", "body")
            )
        else:
            self.model.grad_balance_groups = None
            self.model.grad_balance_shared_params = None
        self.model.conflict_shared_params = get_shared_params(
            self.model, getattr(self.args, "grad_balance_shared", "body")
        )

    def get_validator(self):
        # ✅不要传 pbar，且用关键字保证版本兼容
        return SegPoseValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
        )

    def save_metrics(self, metrics):
        metrics = {**metrics, **self.conflict_metrics}
        super().save_metrics(metrics)

    def _on_train_epoch_end(self, trainer):
        if trainer is not self:
            return
        self._update_conflict_metrics()

    def _update_conflict_metrics(self):
        if RANK not in {-1, 0}:
            return
        shared_params = getattr(self.model, "conflict_shared_params", None)
        if not shared_params:
            return
        try:
            batch = next(iter(self.train_loader))
        except Exception as exc:
            LOGGER.warning(f"conflict metrics skipped: {exc}")
            return
        model = unwrap_model(self.model)
        model.zero_grad(set_to_none=True)
        batch = self.preprocess_batch(batch)
        with autocast(self.amp):
            preds = model(batch["img"])
            loss_bs, _ = model.loss(batch, preds)
        if not hasattr(loss_bs, "numel") or loss_bs.numel() < 6:
            return

        seg_loss = loss_bs[0] + loss_bs[1] + loss_bs[4] + loss_bs[5]
        pose_loss = loss_bs[2] + loss_bs[3]
        det_loss = loss_bs[0] + loss_bs[4] + loss_bs[5]

        grads_seg = torch.autograd.grad(seg_loss, shared_params, retain_graph=True, allow_unused=True)
        grads_pose = torch.autograd.grad(pose_loss, shared_params, retain_graph=True, allow_unused=True)
        grads_det = torch.autograd.grad(det_loss, shared_params, retain_graph=False, allow_unused=True)

        def _pair_metrics(grads_a, grads_b, eps=1e-12):
            device = loss_bs.device
            dot = torch.zeros((), device=device)
            norm_a = torch.zeros((), device=device)
            norm_b = torch.zeros((), device=device)
            neg = torch.zeros((), device=device)
            total = torch.zeros((), device=device)
            for ga, gb in zip(grads_a, grads_b):
                if ga is None or gb is None:
                    continue
                ga = ga.detach().float()
                gb = gb.detach().float()
                dot = dot + (ga * gb).sum()
                norm_a = norm_a + (ga * ga).sum()
                norm_b = norm_b + (gb * gb).sum()
                neg = neg + (ga * gb < 0).sum()
                total = total + ga.new_tensor(ga.numel())
            cos = dot / (torch.sqrt(norm_a) * torch.sqrt(norm_b) + eps)
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            angle = torch.rad2deg(torch.acos(cos))
            neg_ratio = neg / (total + eps)
            return float(cos), float(angle), float(neg_ratio)

        seg_pose = _pair_metrics(grads_seg, grads_pose)
        det_seg = _pair_metrics(grads_det, grads_seg)
        det_pose = _pair_metrics(grads_det, grads_pose)

        self.conflict_metrics["conflict/seg_pose_cos"] = seg_pose[0]
        self.conflict_metrics["conflict/seg_pose_angle"] = seg_pose[1]
        self.conflict_metrics["conflict/seg_pose_neg"] = seg_pose[2]
        self.conflict_metrics["conflict/det_seg_pose_cos"] = (det_seg[0] + det_pose[0] + seg_pose[0]) / 3
        self.conflict_metrics["conflict/det_seg_pose_angle"] = (det_seg[1] + det_pose[1] + seg_pose[1]) / 3
        self.conflict_metrics["conflict/det_seg_pose_neg"] = (det_seg[2] + det_pose[2] + seg_pose[2]) / 3
        model.zero_grad(set_to_none=True)
