import os
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

import streamlit as st

model_path = os.path.join('.', 'last.pt')
print(model_path)
model = YOLO(model_path)  # load a custom model
print(model)
threshold = 0.6

st.set_page_config(
    page_title="Retail product detection", layout="wide"
)

st.sidebar.write("Select from below options")
side = st.sidebar.selectbox(
    "Selcect one", ["Dashboard"]
)

if side == 'Dashboard':
    st.markdown('## Retail product detection')
    st.markdown('### Upload an image below ')
    # Upload the image from streamlit interface
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        img = Image.open(uploaded_file)
        img = np.array(img)  # Convert PIL image to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.imread(uploaded_file)
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

        nums = f'Number of colgates in the image are: {count}'
        st.markdown(nums)

        st.image(resized_img, caption='Predictions on the image', use_column_width=True)

