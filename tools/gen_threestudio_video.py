import cv2
from pathlib import Path

# Define parameters
fps = 30
video_duration = 5  # seconds
frame_size = (256, 256)  # You can adjust this to match the actual size of your images
output_file = "output_video.mp4"

# Get images
images_dir = Path(
    "/home/and/projects/itmo/diploma/libs/threestudio/outputs/dreamfusion-sd/01_Wing_chair_from_IKEA_called_Vibberbo._Furniture_type:_armchair._High_back_for_extra_neck_support.@20240521-155118/save/it10000-test"
)
images_paths = sorted(list(images_dir.iterdir()), key=lambda p: int(p.stem))
images = [
    cv2.imread(str(img_p))[50:452, 50:452] for img_p in images_paths
]  # prolificdreamer dreamfusion
# images = [cv2.imread(str(img_p))[:512, :512] for img_p in images_paths]  # mvdream

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

# Add images to the video
for image in images:
    image = cv2.resize(image, frame_size)
    video_writer.write(image)

# Release the video writer
video_writer.release()

print(f"Video saved as {output_file}")
