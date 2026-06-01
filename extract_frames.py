import cv2
import os

# VIDEO FOLDER
video_folder = r"C:\Users\monic\OneDrive\Desktop\socila media\videos"

# OUTPUT FOLDER
output_folder = r"C:\Users\monic\OneDrive\Desktop\socila media\extracted_frames"

os.makedirs(output_folder, exist_ok=True)

# Get all videos
video_files = [f for f in os.listdir(video_folder)
               if f.endswith(('.mp4', '.mkv', '.avi'))]

print("Videos found:", len(video_files))

# TOTAL frames needed
TOTAL_REQUIRED_FRAMES = 700

saved_count = 0

# Loop through videos
for video_name in video_files:

    video_path = os.path.join(video_folder, video_name)

    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # Save every 30th frame
        if frame_count % 30 == 0:

            # Resize frame
            frame = cv2.resize(frame, (1280, 720))

            # Frame name
            frame_name = f"frame_{saved_count}.jpg"

            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)

            saved_count += 1

            print(f"Saved Frames: {saved_count}", end="\r")

            # STOP when 700 reached
            if saved_count >= TOTAL_REQUIRED_FRAMES:
                break

    cap.release()

    # Stop outer loop too
    if saved_count >= TOTAL_REQUIRED_FRAMES:
        break

print("\n\nDONE")
print("Total Frames Saved:", saved_count)