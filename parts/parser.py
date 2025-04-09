import cv2
import os

image_folder = r'videos/points'
video_name = r'videos/output_points.mp4'
fps = 30

# Collect all image files and sort them
images = sorted([
    img for img in os.listdir(image_folder)
    if img.endswith(".jpg")
])

# Load the first image to get dimensions
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Write each frame
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    video.write(frame)

video.release()
print(f"[✔] Saved video to {video_name}")
