import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

# MediaPipe modules are used for precise human-centric perception.
# FaceMesh gives dense facial landmarks useful for gaze and head pose.
# Hands gives detailed hand landmarks, which are more reliable than box-based detection.
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize face mesh to track a single face with refined landmarks.
# refine_landmarks enables iris and eye contours, useful for later extensions.
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize hand tracking.
# This detects up to two hands and maintains identity across frames.
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# YOLO is used for object-level perception.
# Unlike MediaPipe, YOLO detects semantic objects such as phones.
# The lightweight YOLOv8n model is chosen for real-time performance.
yolo = YOLO("yolov8n.pt")

# COCO dataset class ID for cell phones.
# Used to filter YOLO detections.
PHONE_CLASS_ID = 67

# Camera initialization is handled defensively.
# macOS camera indices are unstable due to Continuity Camera and permissions.
# This loop attempts multiple indices and selects the first one that actually returns frames.
cap = None
for i in range(4):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        ret, _ = test_cap.read()
        if ret:
            cap = test_cap
            print(f"Using camera index {i}")
            break
        test_cap.release()

# If no camera can be opened, the program exits cleanly.
if cap is None:
    print("Error: no working camera found")
    exit()

# These variables store timestamps marking when unsafe behaviors begin.
# Behavior is only flagged if it persists over time, reducing false positives.
hands_on_wheel_start = None
phone_usage_start = None

# Time thresholds (in seconds) required before triggering alerts.
HAND_ON_WHEEL_TIME = 1.0
PHONE_USAGE_TIME = 1.0

# This function checks whether two axis-aligned bounding boxes overlap.
# It is used to infer physical interaction between hands and the steering wheel.
def bbox_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xA < xB and yA < yB

# Computes Euclidean distance between two points.
# Used to estimate proximity between the face and detected phone.
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

print("Starting Driver Monitor... Press 'q' to quit")

while True:
    # Capture a frame from the camera.
    ret, frame = cap.read()
    if not ret:
        continue

    # Mirror the frame for a more natural user-facing display.
    frame = cv2.flip(frame, 1)

    # Cache frame dimensions for landmark scaling.
    h, w, _ = frame.shape

    # MediaPipe expects RGB input instead of OpenCV’s default BGR.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run face and hand landmark inference.
    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    # Run YOLO object detection on the current frame.
    # The confidence threshold filters weak detections.
    yolo_results = yolo(frame, conf=0.4, verbose=False)[0]

    # steering_wheel_box is a heuristic approximation.
    # In production, this would be replaced with a trained wheel detector.
    steering_wheel_box = None
    phone_boxes = []

    # Iterate over all detected objects.
    for box in yolo_results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # If the detected object is a phone, record its bounding box
        # and draw it for visualization.
        if cls == PHONE_CLASS_ID:
            phone_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "PHONE", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Steering wheel detection is approximated using spatial heuristics.
        # Objects appearing low in the frame are assumed to be near the wheel region.
        if steering_wheel_box is None and y2 > h * 0.6:
            steering_wheel_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "WHEEL", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Hand landmarks are converted into bounding boxes for spatial reasoning.
    # This bridges precise pose estimation with object-level overlap logic.
    hand_boxes = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]
            hand_boxes.append(
                (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            )

    # Face landmarks are used to estimate the face center.
    # The nose tip is a stable reference point for proximity checks.
    face_center = None
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame, face_landmarks, mp_face.FACEMESH_TESSELATION
            )
            nose = face_landmarks.landmark[1]
            face_center = (int(nose.x * w), int(nose.y * h))

    now = time.time()

    # Hands-on-wheel logic checks whether any hand overlaps the wheel region.
    # Overlap implies physical contact with the steering wheel.
    hands_on_wheel = False
    if steering_wheel_box:
        for hb in hand_boxes:
            if bbox_overlap(hb, steering_wheel_box):
                hands_on_wheel = True
                break

    # If hands are off the wheel, start timing the unsafe interval.
    # If hands return, reset the timer.
    if hands_on_wheel:
        hands_on_wheel_start = None
    else:
        if hands_on_wheel_start is None:
            hands_on_wheel_start = now

    # Phone usage is inferred by spatial proximity between the face and phone.
    # This avoids relying on hand-only heuristics.
    phone_use = False
    if face_center:
        for pb in phone_boxes:
            px = (pb[0] + pb[2]) // 2
            py = (pb[1] + pb[3]) // 2
            if dist(face_center, (px, py)) < 120:
                phone_use = True

    # Phone usage is also time-gated to reduce transient false detections.
    if phone_use:
        if phone_usage_start is None:
            phone_usage_start = now
    else:
        phone_usage_start = None

    # Alerts are rendered only when unsafe behavior persists past thresholds.
    y = 30

    if hands_on_wheel_start and now - hands_on_wheel_start > HAND_ON_WHEEL_TIME:
        cv2.putText(frame, "HANDS OFF WHEEL",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        y += 40

    if phone_usage_start and now - phone_usage_start > PHONE_USAGE_TIME:
        cv2.putText(frame, "PHONE USAGE DETECTED",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Driver Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
