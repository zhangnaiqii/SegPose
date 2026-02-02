import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO



if __name__ == "__main__":
    # Build seg+pose model from yaml (no pretrained weights).
    model = YOLO("ultralytics/cfg/models/12/segpose.yaml", task='segpose')
    # model.load('yolov12n-seg.pt')



    model.train(
        data="ultralytics/cfg/datasets/straw.yaml",
        epochs=100,
        batch=64,nbs=64,
        workers=24,
        grad_balance='DAGRBalancer',
        grad_balance_groups='segpose3',
        grad_balance_shared=[8],
        name="straw_dagr_neck8",
    )

