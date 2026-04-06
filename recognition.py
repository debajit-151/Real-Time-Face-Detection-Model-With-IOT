import numpy as np

class FaceRecognizer:
    def __init__(self, encoder, threshold=0.40):
        """
        encoder: Instance of FaceEncoder to access the DB
        threshold: The maximum cosine distance to be considered a match. 
                   DeepFace's recommended threshold for Facenet + Cosine is 0.40.
        """
        self.encoder = encoder
        self.threshold = threshold

    def cosine_distance(self, embedding1, embedding2):
        """Compute the cosine distance between two representations"""
        a = np.array(embedding1)
        b = np.array(embedding2)
        
        # Calculate dot product
        dot_product = np.dot(a, b)
        
        # Calculate magnitudes
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Calculate cosine similarity and then distance
        cos_similarity = dot_product / (norm_a * norm_b)
        
        return 1.0 - cos_similarity

    def identify(self, face_img):
        """
        Given a cropped face image, returns the matching name or 'Unknown'.
        """
        db = self.encoder.db
        if not db:
            return "Unknown" # Base is empty
            
        test_embedding = self.encoder.get_embedding(face_img)
        if test_embedding is None:
            return "Unknown"
            
        min_distance = float('inf')
        matched_name = "Unknown"
        
        # Compare with everyone in the database
        for name, embeddings_list in db.items():
            for db_emb in embeddings_list:
                dist = self.cosine_distance(test_embedding, db_emb)
                if dist < min_distance:
                    min_distance = dist
                    if min_distance <= self.threshold:
                        matched_name = name
                        
        return matched_name
