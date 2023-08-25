from ultralytics import YOLO
import multiprocessing as mp

def main():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model using yolov8 nano base model

    # Use the model
    results = model.train(data="config.yaml", epochs=100)  # train the model

if __name__ == '__main__':
    mp.freeze_support()
    main()