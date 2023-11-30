import os

import cv2
import numpy as np


def stack_videos(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frames = []
    print("Press 's' to mark the beginning and end of intervals. Press 'q' to quit.")

    video_frames_1 = []
    video_frames_2 = []

    while len(start_frames) < 4:
        ret, frame = cap.read()
        if not ret:
            break

        if len(start_frames) == 1:
            video_frames_1.append(frame)
        elif len(start_frames) == 3:
            video_frames_2.append(frame)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0)

        if key == ord('s'):
            start_frames.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            print(f"Marked frame {start_frames[-1]}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

    # Extract intervals
    if len(start_frames) < 4:
        print("Not enough intervals marked.")
        return

    # Cut and stack videos
    length_1 = start_frames[1] - start_frames[0]
    length_2 = start_frames[3] - start_frames[2]
    length = max(length_1, length_2)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height * 2))
    for i in range(length):
        idx1 = i % length_1
        idx2 = i % length_2
        frame1 = video_frames_1[idx1]
        frame2 = video_frames_2[idx2]
        frame = np.concatenate([frame1, frame2], axis=0)
        out.write(frame)

    out.release()
    print("Video processing complete.")


# Replace 'path_to_video.mp4' with your video file path
video_dir = '/home/zwyan/3d_cv/papers/my papers/anti-aliasing/Submissions'
input_video_path = os.path.join(video_dir, 'MSGS Supp Video.mp4')
output_video_path = os.path.join(video_dir, 'MSGS_stack.mp4')
stack_videos(input_video_path, output_video_path)
