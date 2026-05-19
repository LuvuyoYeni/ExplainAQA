import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_landmarks = []

def on_result(result, image, timestamp):
    global latest_landmarks
    if result.pose_landmarks:
        latest_landmarks = result.pose_landmarks[0]
    else:
        latest_landmarks = []

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result
)

connections = [
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (11, 12),
    (11, 23),
    (12, 24),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

cap = cv2.VideoCapture(0)
timestamp = 0

left_angles = []
right_angles = []

recording = False
current_rep = []
all_reps = []

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        if len(latest_landmarks) > 0:
            h, w, _ = frame.shape

            for landmark in latest_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            for start, end in connections:
                x1 = int(latest_landmarks[start].x * w)
                y1 = int(latest_landmarks[start].y * h)
                x2 = int(latest_landmarks[end].x * w)
                y2 = int(latest_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            def lm(i):
                return [latest_landmarks[i].x, latest_landmarks[i].y]

            angle_left_knee  = calculate_angle(lm(23), lm(25), lm(27))
            angle_right_knee = calculate_angle(lm(24), lm(26), lm(28))
            angle_left_hip   = calculate_angle(lm(11), lm(23), lm(25))
            angle_right_hip  = calculate_angle(lm(12), lm(24), lm(26))
            angle_left_ankle = calculate_angle(lm(25), lm(27), lm(31))
            angle_right_ankle= calculate_angle(lm(26), lm(28), lm(32))
            angle_trunk      = calculate_angle(lm(11), lm(23), lm(24))

            cv2.putText(frame, str(int(angle_left_knee)),
                        (int(latest_landmarks[25].x * w), int(latest_landmarks[25].y * h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, str(int(angle_right_knee)),
                (int(latest_landmarks[26].x * w), int(latest_landmarks[26].y * h)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if recording:
                current_rep.append([
                    angle_left_knee, angle_right_knee,
                    angle_left_hip,  angle_right_hip,
                    angle_left_ankle, angle_right_ankle,
                    angle_trunk
                ])

            left_angles.append(angle_left_knee)
            right_angles.append(angle_right_knee)

            cv2.putText(frame,f"Frames;{len(left_angles)}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        cv2.imshow("Pose", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            if not recording:
                recording = True
                current_rep = []
                print("recording started")
            else:
                recording = False
                all_reps.append(current_rep)
                print(f"Rep saved. Total reps recorded:{len(all_reps)}")


import json
with open('data/sample_reps.json', 'w') as f:
    json.dump(all_reps, f)
print(f"Saved {len(all_reps)} reps to data/sample_reps.json")

cap.release()
cv2.destroyAllWindows()
