import torch

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


class SegPosePredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segpose"

    @staticmethod
    def _extract_proto(preds):
        p1 = preds[1]
        if torch.is_tensor(p1):
            return p1
        if isinstance(p1, (tuple, list)):
            # prefer a 4D tensor
            for j in (-1, 2, 1, 0):
                if -len(p1) <= j < len(p1) and torch.is_tensor(p1[j]) and p1[j].ndim == 4:
                    return p1[j]
        raise TypeError(f"Invalid preds[1] for proto: {type(p1)}")

    def postprocess(self, preds, img, orig_imgs):
        p = nms.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        proto = self._extract_proto(preds)

        # Unwrap model to locate the actual head module.
        m = self.model
        for _ in range(8):
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
            raise TypeError(f"SegPosePredictor.postprocess: cannot locate head from model type {type(self.model)}")
        nm = int(getattr(head, "nm"))
        kpt_shape = getattr(head, "kpt_shape")
        nk = int(getattr(head, "nk", kpt_shape[0] * kpt_shape[1]))

        results = []
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):
                results.append(
                    Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=None, keypoints=None)
                )
                continue

            coeff = pred[:, 6 : 6 + nm]
            kpts_flat = pred[:, 6 + nm : 6 + nm + nk]

            if self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], coeff, pred[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], coeff, pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            kpts = kpts_flat.view(-1, *kpt_shape) if kpts_flat.numel() else None
            if kpts is not None:
                kpts_xy = kpts[..., :2]
                kpts[..., :2] = ops.scale_coords(img.shape[2:], kpts_xy, orig_img.shape)

            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks, keypoints=kpts))

        return results
