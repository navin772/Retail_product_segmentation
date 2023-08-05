import cv2
import numpy as np
import torch

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # detects cpu or gpu device
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, device=device)

def main():

    # Load the video
    video_path = "cctv.mp4" # change input video path
    cap = cv2.VideoCapture(video_path)

    # Get video details and initialize the heatmap
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

    # Process each frame in the video and accumulate the heatmap for detected persons
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons in the frame
        results = model(frame[:, :, ::-1])  # YOLOv5 takes BGR images, so reverse RGB order
        detections = results.pandas().xyxy[0]
        detections = detections[detections['name'] == 'person']

        # Extract detected person positions and update heatmap
        for _, row in detections.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            heatmap[ymin:ymax, xmin:xmax] += 1

        frame_count += 1

    # Normalize the heatmap based on the number of frames
    heatmap /= frame_count

    # Close the video and reopen it to start from the beginning
    cap.release()
    cap = cv2.VideoCapture(video_path)

    # Process each frame again to overlay the normalized heatmap and save it to a new video file
    output_path = "heatmap_video.mp4"   # Write video output to this path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the current frame with heatmap overlay
        heatmap_normalized = np.uint8(255 * heatmap / heatmap.max())
        heatmap_overlay = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)
        heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_overlay, 0.7, 0)
        out.write(heatmap_overlay)

        # Press 'q' to exit the loop and close the window
        cv2.imshow("Heatmap", heatmap_overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()