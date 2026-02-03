import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO



if __name__ == "__main__":
    # Build seg+pose model from yaml (no pretrained weights).
    model = YOLO("ultralytics/cfg/models/12/segpose.yaml", task='segpose')
    # model = YOLO('runs/segment/baseline100/weights/last.pt', task='segpose')
    # model.load('yolov12n-seg.pt')


    model.train(
        data="ultralytics/cfg/datasets/straw.yaml",
        # resume=True,
        epochs=100,
        batch=64,nbs=64,
        workers=16,
        grad_balance='PCGradBalancer',
        # grad_balance=None,
        grad_balance_shared=['body'],
        name="pc",
    )
