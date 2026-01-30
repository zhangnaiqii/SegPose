import numpy as np
import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import OKS_SIGMA, get_oks_sigma, SegPoseMetrics, kpt_iou



class SegPoseValidator(SegmentationValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=DEFAULT_CFG, _callbacks=None):
        # ✅注意：不把 pbar 传给 super（你的版本不支持）
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.args.task = "segpose"

    def init_metrics(self, model):
        self.model = model  # ✅关键，否则 postprocess 里 self.model 不存在会炸
        super().init_metrics(model)

        # ✅替换 metrics，让输出支持 pose
        self.metrics = SegPoseMetrics(names=self.names)
        self.stats = self.metrics.stats  # ✅保证 update_metrics 使用同一个 stats 容器

    def get_desc(self) -> str:
        # Class + Images + Instances + Box(4) + Mask(4) + Pose(4)
        return ("%22s" + "%11s" * 14) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        # preds: (y, (feats, mc, proto, kpt))  or (y, (feats, mc, proto))
        aux = preds[1]
        if isinstance(aux, (tuple, list)):
            # proto 固定在 index=2
            proto = aux[2]
        else:
            proto = aux

        preds = super(SegmentationValidator, self).postprocess(preds[0])  # 调 DetectionValidator.postprocess
        imgsz = [4 * x for x in proto.shape[2:]]

        # ✅鲁棒取 head：剥掉外层 wrapper，直到拿到 BaseModel，其 .model 才是可下标的层列表
        m = self.model
        for _ in range(8):  # 防御性上限，避免奇葩循环引用
            inner = getattr(m, "model", None)
            if inner is None:
                break
            if isinstance(inner, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
                break
            m = inner

        if hasattr(m, "model") and isinstance(m.model, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
            head = m.model[-1]
        elif isinstance(m, (list, tuple, torch.nn.ModuleList, torch.nn.Sequential)):
            head = m[-1]
        else:
            raise TypeError(f"SegPoseValidator.postprocess: cannot locate head from model type {type(self.model)}")

        nm = int(getattr(head, "nm"))
        kpt_shape = getattr(head, "kpt_shape")
        if kpt_shape is None:
            raise AttributeError("SegPoseValidator.postprocess: head.kpt_shape is None (wrong head picked)")
        nk = int(getattr(head, "nk", kpt_shape[0] * kpt_shape[1]))

        for i, pred in enumerate(preds):
            if len(pred["bboxes"]) == 0:
                pred["masks"] = torch.zeros((0, *proto.shape[2:]), dtype=torch.uint8, device=proto.device)
                pred["keypoints"] = torch.zeros((0, *kpt_shape), device=proto.device)
                continue

            extra = pred.pop("extra")  # (n, nm+nk)
            coefficient = extra[:, :nm]
            kpts_flat = extra[:, nm: nm + nk]

            pred["masks"] = (
                self.process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if self.process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
            pred["keypoints"] = kpts_flat.view(-1, *kpt_shape) if kpts_flat.numel() else kpts_flat.view(0, *kpt_shape)

        return preds

    def update_metrics(self, preds, batch):
        # 先走原版逻辑，产出 box/mask 的 tp, tp_m, conf, pred_cls 等
        super().update_metrics(preds, batch)

        # 然后补 pose 的 tp_p（否则 SegPoseMetrics.process() 只能全 0）
        if "tp_p" not in self.stats:
            raise KeyError("self.stats 缺少 'tp_p'，说明你没有用 SegPoseMetrics 或 init_metrics 没生效。")
        for k in ("batch_idx", "cls", "bboxes", "keypoints", "img"):
            if k not in batch:
                raise KeyError(f"batch 缺少关键字段 '{k}'，无法计算 pose 指标。")

        batch_idx = batch["batch_idx"].view(-1).long()
        gt_cls_all = batch["cls"].view(-1).long()
        gt_box_all = batch["bboxes"]
        gt_kpt_all = batch["keypoints"]

        _, _, h, w = batch["img"].shape
        iouv = torch.as_tensor(self.iouv, device=gt_box_all.device).view(-1)

        for si, pred in enumerate(preds):
            if "keypoints" not in pred:
                raise KeyError("pred 缺少 'keypoints'，说明 postprocess 没把关键点塞回 preds。")
            if "cls" not in pred:
                raise KeyError("pred 缺少 'cls'，无法做按类匹配计算 pose TP。")

            pcls = pred["cls"].view(-1).long()
            pkpt = pred["keypoints"]

            gi = batch_idx == si
            gcls = gt_cls_all[gi]
            gbox = gt_box_all[gi]
            gkpt = gt_kpt_all[gi]

            if gkpt.numel() == 0:
                valid_gt = gcls.new_zeros((0,), dtype=torch.bool)
            elif gkpt.shape[-1] == 3:
                valid_gt = (gkpt[..., 2] != 0).any(dim=1)
            else:
                valid_gt = gcls.new_ones((gkpt.shape[0],), dtype=torch.bool)

            gcls_pose = gcls[valid_gt]
            if "target_cls_p" not in self.stats:
                self.stats["target_cls_p"] = []
            self.stats["target_cls_p"].append(gcls_pose.cpu().numpy())

            tp_p = self._compute_tp_p(
                pred_kpts=pkpt,
                pred_cls=pcls,
                gt_kpts=gkpt[valid_gt] if gkpt.numel() else gkpt,
                gt_cls=gcls_pose,
                gt_bboxes=gbox[valid_gt] if gbox.numel() else gbox,
                img_hw=(h, w),
                iouv=iouv,
            )
            self.stats["tp_p"].append(tp_p)

    @staticmethod
    def _ensure_kpt_dim3(kpts: torch.Tensor) -> torch.Tensor:
        # (N,K,2)->(N,K,3) 让 kpt_iou 不用猜
        if kpts.numel() == 0:
            return kpts
        if kpts.shape[-1] == 3:
            return kpts
        if kpts.shape[-1] == 2:
            ones = torch.ones((*kpts.shape[:-1], 1), device=kpts.device, dtype=kpts.dtype)
            return torch.cat([kpts, ones], dim=-1)
        raise ValueError(f"Unsupported keypoints last-dim={int(kpts.shape[-1])}, expected 2 or 3.")

    @staticmethod
    def _scale_xywh_if_normalized(xywh: torch.Tensor, img_hw: tuple[int, int]) -> torch.Tensor:
        # 只要看最大值就够了：<=1 基本就是归一化；>1 就当像素
        if xywh.numel() == 0:
            return xywh
        h, w = img_hw
        if float(xywh.max()) <= 1.01:
            gain = xywh.new_tensor([w, h, w, h])
            return xywh * gain
        return xywh

    @staticmethod
    def _scale_kpts_if_normalized(kpts: torch.Tensor, img_hw: tuple[int, int]) -> torch.Tensor:
        if kpts.numel() == 0:
            return kpts
        h, w = img_hw
        # 注意：只看 xy，两维通常是 0~1；可见性维可能是 2
        if float(kpts[..., :2].max()) <= 1.01:
            kpts = kpts.clone()
            kpts[..., 0] *= w
            kpts[..., 1] *= h
        return kpts

    def _compute_tp_p(
            self,
            pred_kpts: torch.Tensor,  # (N, K, 2/3)
            pred_cls: torch.Tensor,  # (N,)
            gt_kpts: torch.Tensor,  # (M, K, 2/3)
            gt_cls: torch.Tensor,  # (M,)
            gt_bboxes: torch.Tensor,  # (M, 4)  xywh (norm 或 pixel)
            img_hw: tuple[int, int],  # (h, w)
            iouv: torch.Tensor,  # (T,)
    ) -> np.ndarray:
        n_pred = int(pred_kpts.shape[0])
        t = int(iouv.numel())
        if n_pred == 0:
            return np.zeros((0, t), dtype=bool)

        pred_kpts = self._ensure_kpt_dim3(pred_kpts)
        gt_kpts = self._ensure_kpt_dim3(gt_kpts)

        pred_kpts = self._scale_kpts_if_normalized(pred_kpts, img_hw)
        gt_kpts = self._scale_kpts_if_normalized(gt_kpts, img_hw)

        if gt_kpts is None or int(gt_kpts.shape[0]) == 0:
            return np.zeros((n_pred, t), dtype=bool)

        gt_bboxes = self._scale_xywh_if_normalized(gt_bboxes, img_hw)
        gt_xyxy = ops.xywh2xyxy(gt_bboxes)
        area = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp_(0) * (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp_(0)

        # ✅ sigma 长度必须等于 K；优先用 data.yaml 里给的 kpt_sigma（如果你加了）
        kpt_shape = (int(gt_kpts.shape[1]), int(gt_kpts.shape[2]))
        sigma = get_oks_sigma(kpt_shape=kpt_shape, kpt_sigma=self.data.get("kpt_sigma", None))

        oks = kpt_iou(gt_kpts, pred_kpts, area=area, sigma=sigma.tolist())  # (M, N)
        oks *= (gt_cls.view(-1, 1) == pred_cls.view(1, -1)).to(oks.device)

        oks_np = oks.detach().cpu().numpy()
        tp_p = np.zeros((n_pred, t), dtype=bool)
        iouv_list = iouv.detach().cpu().tolist()

        for ti, thr in enumerate(iouv_list):
            x = np.where(oks_np >= thr)
            if x[0].size == 0:
                continue
            matches = np.concatenate((np.stack(x, 1), oks_np[x[0], x[1]][:, None]), 1)  # [gt, pred, oks]
            if matches.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            tp_p[matches[:, 1].astype(int), ti] = True

        return tp_p

