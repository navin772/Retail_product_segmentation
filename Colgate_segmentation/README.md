# Colgate toothpaste segmentation
## Introduction
The ML model is trained on [YOLOv8](https://github.com/ultralytics/ultralytics) using a custom dataset of individual colgates and on shelves.

### Dataset
The dataset is created using images downloaded from google images using the `image_downloader.py` python script. The script uses the `icrawler` library to download images from google images. Some additional shelf images where found from the [SKU110K Dataset](https://www.kaggle.com/datasets/thedatasith/sku110k-annotations) on kaggle.

### Annotations and bounding boxes
The dataset was labelled and annotated using [CVAT](cvat.ai) and the annotations were exported as YOLO format. The annotations are stored in the `Model_training/labels/train` folder. The annotations are in text(.txt) format.

### Training
1. The model was trained using the YOLOv8 nano model as the base model on the custom dataset we created and annotated earlier.
2. To use a GPU for training (recommended), use the `gpu_detection.py` script to check whether pytorch is configured correctly to use the GPU.
3. Create the `config.yaml` file that is used by YOLO to specify the dataset path. Configure the values correctly in it.
4. Run the `train.py` script to start training the model. The output will be saved in the `runs/detect/train/` folder.
5. The model with the .pt extension will be saved in the `runs/detect/train/weights/` folder.

### Inference
1. The `detect.py` script is used to run inference on the test images.
2. The `test_images` directory contains the test images on which the inference can be run, you can also use your own images. Make sure to add it to the `test_images` directory and mention it's path in the `detect.py` script.