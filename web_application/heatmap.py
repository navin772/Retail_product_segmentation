import os
import cv2
import numpy as np
import streamlit as st
import torch


def generate_heatmap(video_path):

    # Load the model
    device = torch.device('cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device=device)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame[:, :, ::-1])
        detections = results.pandas().xyxy[0]
        detections = detections[detections['name'] == 'person']

        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            heatmap[ymin:ymax, xmin:xmax] += 1

        frame_count += 1

    heatmap /= frame_count
    cap.release()
    cap = cv2.VideoCapture(video_path)

    output_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        heatmap_normalized = np.uint8(255 * heatmap / heatmap.max())
        heatmap_overlay = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
        heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.7, 0)
        output_frames.append(heatmap_overlay)

    cap.release()

    return output_frames, frame_rate


def main():
    
    st.title("Video Heatmap Application")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_file is not None:
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.text("Generating heatmap...")
        output_frames, frame_rate = generate_heatmap(video_path)

        st.text("Creating heatmap video...")

        st.text("This may take a few minutes...")
        
        # Create a temporary directory to hold video frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        # Save the frames as images in the temporary directory
        for i, frame in enumerate(output_frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)

        # Create the final video using ffmpeg
        output_path = "output_heatmap_video.mp4"

        if os.path.exists(output_path):
            os.remove(output_path)  

        ffmpeg_cmd = (
            f"ffmpeg -framerate {frame_rate} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_path}"
        )
        os.system(ffmpeg_cmd)

        # Remove the temporary frames directory
        os.system(f"rm -rf {temp_dir}")

        st.text("Heatmap video created!")
        st.video(output_path)

