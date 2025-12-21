import cv2

# Use AVFoundation for macOS
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera not opened!")
    exit()

cv2.namedWindow("Webcam Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not grabbed, retrying...")
        continue  # keep trying

    cv2.imshow("Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
