import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO





if __name__ == "__main__":
    # Build seg+pose model from yaml (no pretrained weights).
    # model = YOLO("ultralytics/cfg/models/12/csp.yaml", task='segpose')
    model = YOLO('runs/segment/magpc_csp_cos/weights/last.pt', task='segpose')
    # model.load('yolov12n-seg.pt')

    # Z:/datasets/segment/straw_segpose/straw-pose.yaml

    model.train(
        data="ultralytics/cfg/datasets/straw.yaml",
        resume=True,
        epochs=100,
        batch=64,nbs=64,
        workers=16,
        grad_balance='MagPCGradBalancer',
        project='Z:/SegPose/SegPose/runs/segment',
        # grad_balance=None,
        grad_balance_shared=['body'],
        name="magpc_csp_cos",
    )

