from ultralytics import YOLO

model = YOLO('runs/segment/coco7/weights/best.pt')



def main():
    model.val(
        data='ultralytics/cfg/datasets/coco.yaml',
        save=False,
    )

if __name__ == '__main__':
    main()
