import os
from ultralytics import YOLO
import cv2

IMAGE_DIR = os.path.join('.', 'test_images')
image_path = os.path.join(IMAGE_DIR, 'train_592.jpg')
# print(image_path)

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
# print(model_path)

# Load the model
model = YOLO(model_path)  # load a custom model

threshold = 0.6

# Read the image
img = cv2.imread(image_path)
results = model.predict(img, stream=True)

count = 0

for result in results:
    boxes = result.boxes.cpu().numpy() # get boxes on cpu in numpy
    for box in boxes: # iterate boxes
        count += 1
        r = box.xyxy[0].astype(int) # get corner points as int
        print(r) # print boxes
        cv2.rectangle(img, r[:2], r[2:], (0,255,0), 2) # draw boxes on img


scale_percent = 30  # adjust this value to change the scaling factor
new_width = int(img.shape[1] * scale_percent / 100)
new_height = int(img.shape[0] * scale_percent / 100)
resized_img = cv2.resize(img, (new_width, new_height))

print(count)

cv2.imshow('Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
