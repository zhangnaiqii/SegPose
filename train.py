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
        batch=64,
        workers=20,
        grad_balance='DAGR2Balancer',
        grad_balance_groups='segpose3',
        grad_balance_shared=[8],
        name="straw_rgb_neck8_3",
    )

