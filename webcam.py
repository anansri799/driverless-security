import cv2
import mediapipe as mp
import numpy as np

#setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

#webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not access webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb)
    face_results = face.process(rgb)

    h, w, _ = frame.shape

    num_hands = 0

    if hand_results.multi_hand_landmarks:
        num_hands = len(hand_results.multi_hand_landmarks)

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # -----------------------------
    # FACE
    # -----------------------------
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                )
            )

    cv2.putText(
        frame,
        f"Hands detected: {num_hands}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Driver Monitoring Starter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
