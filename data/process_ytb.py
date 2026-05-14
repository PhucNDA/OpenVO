import cv2
import os


# Extract video wrt framerate

def split_video_by_hz_precise(video_path, save_dir, target_hz):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    period = 1.0 / target_hz
    next_t = 0.0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # current time in seconds
        
        if t >= next_t:
            out_path = os.path.join(save_dir, f"{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
            next_t += period

    cap.release()
    print(f"Saved {saved} frames at exactly {target_hz} Hz.")


video_path = '/fs/nexus-projects/AD_dashrecon/ADVO/data/YTB/YTB_driving1.mp4'
save_dir = '/fs/nexus-projects/AD_dashrecon/ADVO/data/YTB/10Hz/sequences/01/image_2'
# measure FPS
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(fps)
split_video_by_hz_precise(video_path, save_dir, 10)