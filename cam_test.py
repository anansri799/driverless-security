import cv2

print("Opening camera 1...")
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Is opened:", cap.isOpened())

if not cap.isOpened():
    print("Error: cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    print("Frame grabbed:", ret)

    if not ret:
        continue

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
