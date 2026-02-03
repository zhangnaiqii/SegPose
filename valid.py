from ultralytics import YOLO

model = YOLO('runs/segment/baseline/weights/best.pt')



def main():
    model.val(
        data='ultralytics/cfg/datasets/straw.yaml',
        save=False,
    )

if __name__ == '__main__':
    main()
