import cv2
import os
from face_detection import FaceDetector

def register_new_person(name, encoder):
    """
    Opens the webcam, instructs the user to center their face, and captures an image
    when 'C' is pressed.
    """
    # Create directory for the person
    person_dir = os.path.join("dataset", name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return False

    detector = FaceDetector()
    print(f"[INFO] Registering {name}. Press 'C' to capture multiple poses. Press 'Q' when finished.")

    capture_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # Detect face primarily to show a bounding box so user knows they are centered
        faces = detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Captured: {capture_count} | 'C' to add, 'Q' to finish", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imshow("Register New Person", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            # Capture if at least 1 face is visible
            if len(faces) > 0:
                # Take the first face
                (x, y, w, h) = faces[0]
                face_img = detector.extract_face(frame, (x, y, w, h))
                
                # Save image
                img_path = os.path.join(person_dir, f"capture_{capture_count}.jpg")
                cv2.imwrite(img_path, frame) # save full frame just in case
                
                # We can also save the cropped face separately if we want
                cropped_path = os.path.join(person_dir, f"face_{capture_count}.jpg")
                cv2.imwrite(cropped_path, face_img)
                
                # Add to DB
                print("[INFO] Computing embeddings... Please wait.")
                success = encoder.add_person(name, face_img)
                if success:
                    capture_count += 1
                    print(f"[SUCCESS] {name} has been successfully registered (capture {capture_count}).")
                else:
                    print(f"[ERROR] Failed to extract facial features for {name}.")
                    
                # We DO NOT break here anymore, allow multiple captures!
            else:
                print("[WARNING] No face detected! Please ensure your face is inside the frame.")
                
        elif key == ord('q') or key == ord('Q'):
            print(f"[INFO] Registration finished. Total faces added to profile: {capture_count}")
            break

    cap.release()
    cv2.destroyAllWindows()
