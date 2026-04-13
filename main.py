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
        self.root.geometry("450x520")
        
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
        
        # ── Modern UI Layout ─────────────────────────────────
        title_label = ctk.CTkLabel(self.root, text="Smart Face Identity Base", 
                                   font=ctk.CTkFont(family="Inter", size=24, weight="bold"))
        title_label.pack(pady=(30, 20))
        
        self.btn_register = ctk.CTkButton(self.root, text="Register New Person", 
                                          font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                          command=self.open_registration_flow, 
                                          fg_color="#10B981", hover_color="#059669",
                                          height=40, width=280, corner_radius=8)
        self.btn_register.pack(pady=(0, 12))
        
        self.btn_recognize = ctk.CTkButton(self.root, text="Launch Real-Time Recognition", 
                                           font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                           command=self.start_recognition, 
                                           fg_color="#3B82F6", hover_color="#2563EB",
                                           height=40, width=280, corner_radius=8)
        self.btn_recognize.pack(pady=12)
        
        # ── Hardware Controls Section ─────────────────────────
        separator = ctk.CTkLabel(self.root, text="── Hardware Controls ──",
                                 font=ctk.CTkFont(family="Inter", size=11),
                                 text_color="#888888")
        separator.pack(pady=(18, 8))

        # PIR Toggle
        self.pir_state_text = ctk.StringVar(value="PIR Sensor: ON")
        self.btn_pir_toggle = ctk.CTkButton(
            self.root, textvariable=self.pir_state_text,
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            command=self.toggle_pir,
            fg_color="#8B5CF6", hover_color="#7C3AED",
            height=38, width=280, corner_radius=8
        )
        self.btn_pir_toggle.pack(pady=6)

        # Scanning Toggle
        self.scan_state_text = ctk.StringVar(value="Camera Rotation: ON")
        self.btn_scan_toggle = ctk.CTkButton(
            self.root, textvariable=self.scan_state_text,
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
            command=self.toggle_scanning,
            fg_color="#F59E0B", hover_color="#D97706",
            height=38, width=280, corner_radius=8
        )
        self.btn_scan_toggle.pack(pady=6)

        # Exit
        self.btn_quit = ctk.CTkButton(self.root, text="Exit Application", 
                                      font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
                                      command=self.root.quit, 
                                      fg_color="#EF4444", hover_color="#DC2626",
                                      height=40, width=280, corner_radius=8)
        self.btn_quit.pack(pady=(18, 20))

    # ── Toggle Callbacks ─────────────────────────────────────

    def toggle_pir(self):
        """Toggle PIR sensor via servo controller and update button text."""
        new_state = self.servo.toggle_pir()
        if new_state:
            self.pir_state_text.set("PIR Sensor: ON")
            self.btn_pir_toggle.configure(fg_color="#8B5CF6", hover_color="#7C3AED")
        else:
            self.pir_state_text.set("PIR Sensor: OFF")
            self.btn_pir_toggle.configure(fg_color="#6B7280", hover_color="#4B5563")

    def toggle_scanning(self):
        """Toggle camera scanning/rotation and update button text."""
        new_state = self.servo.toggle_scanning()
        if new_state:
            self.scan_state_text.set("Camera Rotation: ON")
            self.btn_scan_toggle.configure(fg_color="#F59E0B", hover_color="#D97706")
        else:
            self.scan_state_text.set("Camera Rotation: OFF")
            self.btn_scan_toggle.configure(fg_color="#6B7280", hover_color="#4B5563")

    # ── Registration ─────────────────────────────────────────

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

    # ── Recognition Loop ─────────────────────────────────────

    def start_recognition(self):
        # Hide main window
        self.root.withdraw()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.root.deiconify()
            return
            
        print("[INFO] Starting video stream... Press 'Q' to quit.")
        
        # Frame skipping logic — recognition is heavy, detection is cheap
        frame_count = 0
        process_every_n_frames = 3  # Run DeepFace recognition every 3 frames
        
        # Cache for recognized names to display while skipping frames
        current_faces = []
        
        # Face tracking state
        last_face_seen_time = 0
        face_lost_timeout = 2.0  # seconds before resuming scan after losing face
        is_tracking = False
        
        # Tell ESP32 to start scanning (if enabled)
        if self.servo.connected:
            self.servo.send_scan()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            display_frame = frame.copy()
            
            # ── Fast face detection every frame (Haar cascade — cheap) ──
            faces = self.detector.detect_faces(frame)
            frame_height, frame_width = frame.shape[:2]
            
            # ── Heavy recognition only every Nth frame ──────────────
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
            
            # ── Servo Tracking (runs EVERY frame for smoothness) ─────
            if self.servo.connected:
                # Find the largest detected face (by area) from the fast detector
                # This ensures tracking is responsive even on non-recognition frames
                largest_face = None
                largest_area = 0
                
                for (fx, fy, fw, fh) in faces:
                    area = fw * fh
                    if area > largest_area:
                        largest_area = area
                        largest_face = (fx, fy, fw, fh)
                
                if largest_face is not None:
                    last_face_seen_time = time.time()
                    bx, by, bw, bh = largest_face
                    
                    # Use the smoothed angle computation
                    target_angle = self.servo.compute_smoothed_angle(
                        face_x=bx, face_w=bw,
                        frame_width=frame_width
                    )
                    self.servo.send_track(target_angle)
                    is_tracking = True
                else:
                    # No face detected: check if we should resume scanning
                    if is_tracking and (time.time() - last_face_seen_time > face_lost_timeout):
                        self.servo.send_scan()
                        is_tracking = False
                
            # ── Draw bounding boxes and labels ───────────────────
            for (x, y, w, h) in faces:
                # Match this detection to the closest cached recognition result
                name_to_display = "Scanning..."
                for cf in current_faces:
                    cx, cy, cw, ch = cf["bbox"]
                    # Simple spatial proximity matching
                    if abs(x - cx) < 50 and abs(y - cy) < 50:
                        name_to_display = cf["name"]
                        break
                
                # Colors (BGR format in OpenCV)
                if name_to_display == "Spoof!":
                    color = (0, 0, 255)  # Red for fake
                    name_to_display = "SPOOF DETECTED!"
                elif name_to_display not in ["Unknown", "Scanning..."]:
                    color = (0, 255, 0)  # Green for known live person
                else:
                    color = (255, 165, 0) if name_to_display == "Scanning..." else (0, 0, 255)
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_frame, name_to_display, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # ── Servo + PIR Status Overlay ────────────────────────
            if self.servo.connected:
                status_color = (200, 200, 200)
                pir_label = "ON" if self.servo.pir_enabled else "OFF"
                scan_label = "ON" if self.servo.scanning_enabled else "OFF"
                
                cv2.putText(display_frame, 
                            f"Servo: {self.servo.esp_state} | Angle: {self.servo.current_angle}",
                            (10, frame_height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                motion_txt = "Motion: YES" if self.servo.motion_detected else "Motion: NO"
                motion_color = (0, 255, 0) if self.servo.motion_detected else (100, 100, 100)
                cv2.putText(display_frame, motion_txt,
                            (10, frame_height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
                
                cv2.putText(display_frame,
                            f"PIR: {pir_label} | Scan: {scan_label}",
                            (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            cv2.imshow("Real-Time Face Recognition", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.servo.connected:
            self.servo.send_stop()
                
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS fix for window cleanup
        
        # Show main window again
        self.root.deiconify()

if __name__ == "__main__":
    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.mainloop()
