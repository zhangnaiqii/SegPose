from ultralytics import YOLO

model = YOLO('runs/segment/bbrdel/weights/best.pt')



def main():
    model.predict(
        'Z:/datasets/pointCloud/bbr/pose/images/train',
        save=True,
    )

if __name__ == '__main__':
    main()
