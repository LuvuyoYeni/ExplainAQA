# Code Citations

## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```


## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```


## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```


## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```


## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```


## License: unknown
https://github.com/yudhaputraa/Dasar_Pengelolahan_Citra/blob/b051770c597fe097fd9301313f2280da9146333a/blazepose/blazepose_1.py

```
Unfortunately, PyPI only has MediaPipe 0.10.30+ available, which use the new `tasks` API. Here's a working alternative using the current version:

Replace your code with this version that works with MediaPipe 0.10.33:

```python
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat

# Download the pose model from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models
# Save it as 'pose_landmarker.task' in your project folder

BN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255))
TN_DICT = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0))

options = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path='pose_landmarker.task'))
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    image = mp.Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(image)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks[0]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release
```

