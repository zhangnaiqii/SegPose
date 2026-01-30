import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO



if __name__ == "__main__":
    # Build seg+pose model from yaml (no pretrained weights).
    model = YOLO("ultralytics/cfg/models/12/segpose.yaml", task='segpose')
    model.load('yolov12n-seg.pt')


    model.train(
        data="ultralytics/cfg/datasets/coco.yaml",
        epochs=50,
        batch=48,
        workers=16,
        grad_balance='RGBBalancer',
        grad_balance_groups='segpose3',
        grad_balance_shared='body',
        # imgsz=2080,
        name="coco_rgb_3",
    )

