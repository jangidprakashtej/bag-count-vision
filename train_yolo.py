"""
Train a custom YOLOv8 object detector on your labeled bag images.

Prerequisite: a labeled dataset exported in YOLOv8 format (e.g. from
Roboflow or LabelImg), which gives you a folder structure like:

    bag_dataset/
        data.yaml
        train/images/  train/labels/
        valid/images/  valid/labels/
        test/images/   test/labels/  (optional)

data.yaml lists your class names (categories) and points to the image folders.

Install (one-time):
    pip3.11 install ultralytics

Run:
    python3.11 train_yolo.py
"""
from ultralytics import YOLO

DATA_YAML_PATH = "bag_dataset/data.yaml"   # path to your exported dataset's data.yaml
EPOCHS = 50
IMAGE_SIZE = 640
MODEL_SIZE = "yolov8n.pt"   # "nano" -- smallest/fastest, good starting point on a laptop CPU
                             # step up to "yolov8s.pt" for more accuracy if training time allows


def main():
    model = YOLO(MODEL_SIZE)  # loads a pretrained YOLOv8 base, we fine-tune it on your bags

    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        patience=15,     # stop early if it stops improving, saves time
        project="bag_detector_training",
        name="run1",
    )

    # After training, the best weights are saved at:
    #   bag_detector_training/run1/weights/best.pt
    # That file is what main.py will load for live detection.
    print("\nTraining complete. Best weights saved under bag_detector_training/run1/weights/best.pt")


if __name__ == "__main__":
    main()
