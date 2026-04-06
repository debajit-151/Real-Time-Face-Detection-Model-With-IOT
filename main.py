import customtkinter as ctk
from tkinter import simpledialog, messagebox
import cv2
import threading
import time
import os

from face_detection import FaceDetector
from face_encoding import FaceEncoder
from recognition import FaceRecognizer
from add_person import register_new_person
from servo_controller import ServoController
import logger

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
from deepface import DeepFace

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Face Recognition System")
        self.root.geometry("450x380")
        
        # Initialize Core Modules
        print("[INFO] Loading Face Encoder Model (Facenet)... This may take a few seconds.")
        # Make sure tensorflow doesn't print all info logs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        self.encoder = FaceEncoder(model_name="Facenet")
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(encoder=self.encoder, threshold=0.40)
        
        # Initialize Servo Controller (graceful fallback if ESP32 not connected)
        self.servo = ServoController()
        self.servo.connect()  # Auto-detects or silently disables
        
        # Modern UI Elements
        title_label = ctk.CTkLabel(self.root, text="Smart Face Identity Base", 
                                   font=ctk.CTkFont(family="Inter", size=24, weight="bold"))
        title_label.pack(pady=(40, 30))
        
        self.btn_register = ctk.CTkButton(self.root, text="Register New Person", 
                                          font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                          command=self.open_registration_flow, 
                                          fg_color="#10B981", hover_color="#059669",
                                          height=40, width=280, corner_radius=8)
        self.btn_register.pack(pady=(0, 15))
        
        self.btn_recognize = ctk.CTkButton(self.root, text="Launch Real-Time Recognition", 
                                           font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                           command=self.start_recognition, 
                                           fg_color="#3B82F6", hover_color="#2563EB",
                                           height=40, width=280, corner_radius=8)
        self.btn_recognize.pack(pady=15)
        
        self.btn_quit = ctk.CTkButton(self.root, text="Exit Application", 
                                      font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                      command=self.root.quit, 
                                      fg_color="#EF4444", hover_color="#DC2626",
                                      height=40, width=280, corner_radius=8)
        self.btn_quit.pack(pady=(15, 20))

    def open_registration_flow(self):
        # Prompt for name
        name = simpledialog.askstring("Input", "Enter the person's name:", parent=self.root)
        if name:
            # Hide main window to avoid distraction
            self.root.withdraw()
            register_new_person(name.strip(), self.encoder)
            # Show main window again
            self.root.deiconify()
            messagebox.showinfo("Success", f"Registration flow completed for {name}.")

    def start_recognition(self):
        # Hide main window
        self.root.withdraw()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.deiconify()
            return
            
        print("[INFO] Starting video stream... Press 'Q' to quit.")
        
        # Frame skipping logic to keep CPU happy
        frame_count = 0
        process_every_n_frames = 3 # Process recognition every 3 frames
        
        # Cache for recognized names to display while skipping frames
        current_faces = []
        
        # Face tracking state
        last_face_seen_time = 0
        face_lost_timeout = 1.5  # seconds before resuming scan
        is_tracking = False
        
        # Tell ESP32 to start scanning
        if self.servo.connected:
            self.servo.send_scan()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            display_frame = frame.copy()
            
            # Detect faces every frame to keep bounding boxes smooth
            faces = self.detector.detect_faces(frame)
            frame_height, frame_width = frame.shape[:2]
            
            if frame_count % process_every_n_frames == 0:
                current_faces = []
                try:
                    # Run DeepFace extraction with liveness detection
                    df_faces = DeepFace.extract_faces(frame, anti_spoofing=True, enforce_detection=False)
                    for face_obj in df_faces:
                        if face_obj.get("confidence", 0) > 0.5:
                            area = face_obj["facial_area"]
                            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
                            
                            is_real = face_obj.get("is_real", True)
                            
                            if is_real:
                                face_bgr = self.detector.extract_face(frame, (x, y, w, h))
                                name = self.recognizer.identify(face_bgr)
                            else:
                                name = "Spoof!"
                                
                            current_faces.append({"bbox": (x, y, w, h), "name": name})
                            if name not in ["Unknown", "Spoof!"]:
                                logger.log_recognition(name)
                except Exception as e:
                    pass
                
                # ── Servo Tracking Logic ─────────────────────────
                if self.servo.connected:
                    # Find the first recognized (non-unknown, non-spoof) face to track
                    tracked_face = None
                    for cf in current_faces:
                        if cf["name"] not in ["Unknown", "Spoof!", "Scanning..."]:
                            tracked_face = cf
                            break
                    
                    if tracked_face:
                        last_face_seen_time = time.time()
                        bx, by, bw, bh = tracked_face["bbox"]
                        target_angle = ServoController.face_center_to_angle(
                            face_x=bx, face_w=bw,
                            frame_width=frame_width,
                            current_angle=self.servo.current_angle
                        )
                        self.servo.send_track(target_angle)
                        is_tracking = True
                    else:
                        # No recognized face: check if we should resume scanning
                        if is_tracking and (time.time() - last_face_seen_time > face_lost_timeout):
                            self.servo.send_scan()
                            is_tracking = False
            else:
                # If we skipped recognition, try to match current bounding boxes to the cached faces
                # by simple spatial proximity (basic tracking). For simplicity, we just use the detection
                # but might lose labels for 1-2 frames. Here we just draw boxes.
                pass
                
            # Draw
            for (x, y, w, h) in faces:
                # Find matching name from current_faces by overlap or just simple matching
                # Since we only process every n frames, tracking identity across bounding boxes without a tracker
                # can be tricky. A simple fallback is to just display the last known names.
                name_to_display = "Scanning..."
                for cf in current_faces:
                    cx, cy, cw, ch = cf["bbox"]
                    # Calculate IoU or distance to match
                    if abs(x - cx) < 50 and abs(y - cy) < 50:
                        name_to_display = cf["name"]
                        break
                
                # Colors (BGR format in OpenCV)
                if name_to_display == "Spoof!":
                    color = (0, 0, 255) # Red for fake
                    name_to_display = "SPOOF DETECTED!"
                elif name_to_display not in ["Unknown", "Scanning..."]:
                    color = (0, 255, 0) # Green for known live person
                else:
                    color = (255, 165, 0) if name_to_display == "Scanning..." else (0, 0, 255)
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, name_to_display, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ── Servo Status Overlay ─────────────────────────
            if self.servo.connected:
                status_color = (200, 200, 200)
                cv2.putText(display_frame, f"Servo: {self.servo.esp_state} | Angle: {self.servo.current_angle}",
                            (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                motion_txt = "Motion: YES" if self.servo.motion_detected else "Motion: NO"
                motion_color = (0, 255, 0) if self.servo.motion_detected else (100, 100, 100)
                cv2.putText(display_frame, motion_txt,
                            (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)

            cv2.imshow("Real-Time Face Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.servo.connected:
            self.servo.send_stop()
                
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1) # macOS fix for window cleanup
        
        # Show main window again
        self.root.deiconify()

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.mainloop()
