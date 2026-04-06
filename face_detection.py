import cv2

class FaceDetector:
    def __init__(self):
        # Load the pre-trained Haar cascade classifier for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise IOError("Could not load Haar cascade classifier.")

    def detect_faces(self, frame):
        """
        Detect faces in a given BGR frame.
        Returns a list of bounding boxes: (x, y, w, h)
        """
        # Convert to grayscale for Haar cascade 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # scaleFactor compensates for faces appearing smaller as they are further away
        # minNeighbors specifies how many neighbors each candidate rectangle should have to retain it
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces

    def extract_face(self, frame, bbox):
        """
        Extract the face ROI from the frame given a bounding box.
        bbox format: (x, y, w, h)
        Returns the cropped face image.
        """
        x, y, w, h = bbox
        # Crop the face from the original BGR frame
        cropped_face = frame[y:y+h, x:x+w]
        return cropped_face
