from deepface import DeepFace
import cv2
cap = cv2.VideoCapture(0)
for _ in range(5): cap.read() # warm up
ret, frame = cap.read()
if ret:
    results = DeepFace.extract_faces(frame, anti_spoofing=True, enforce_detection=False)
    for res in results:
        print("is_real:", res.get("is_real"))
        print("score:", res.get("antispoof_score"))
cap.release()
