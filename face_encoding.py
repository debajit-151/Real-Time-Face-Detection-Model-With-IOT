import os
import pickle
import numpy as np
from deepface import DeepFace

DB_PATH = "data/embeddings.pkl"

class FaceEncoder:
    def __init__(self, model_name="Facenet"):
        self.model_name = model_name
        self.db = self.load_db()

    def load_db(self):
        """Loads the embeddings database from disk"""
        if os.path.exists(DB_PATH):
            try:
                with open(DB_PATH, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading database: {e}")
                return {}
        return {}

    def save_db(self):
        """Saves the embeddings database to disk"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with open(DB_PATH, 'wb') as f:
            pickle.dump(self.db, f)

    def get_embedding(self, face_img):
        """
        Uses DeepFace to extract the 128D (or 512D) embedding vector for a given face image.
        face_img: BGR numpy array cropped bounding box of a face.
        """
        try:
            # We set enforce_detection to False because the face is already cropped
            # We don't want deepface to throw an exception if its own detector couldn't find a face
            results = DeepFace.represent(
                img_path=face_img, 
                model_name=self.model_name, 
                enforce_detection=False,
                detector_backend="skip" # We already cropped it
            )
            
            # represent returns a list of dictionaries (one for each face found).
            # Since we cropped it, we expect at least 1.
            if len(results) > 0:
                return np.array(results[0]["embedding"])
            return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def add_person(self, name, face_img):
        """
        Generates embedding for face_img and saves it under 'name' in the DB.
        """
        embedding = self.get_embedding(face_img)
        if embedding is not None:
            if name not in self.db:
                self.db[name] = []
            self.db[name].append(embedding)
            self.save_db()
            print(f"[INFO] Successfully added new embedding for {name}.")
            return True
        return False
