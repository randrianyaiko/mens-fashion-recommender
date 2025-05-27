from src.vector_database.vectorstore import client
from typing import List
from dotenv import load_dotenv
import os
load_dotenv()

class RecommendationEngine:
    def __init__(self):
        self.client = client
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    def get_recommendations(self,liked_images:List[str],disliked_images:List[str],limit: int = 10) -> List[dict]:
        """
        Get recommended images based on liked and disliked images.
        
        Args:
            limit (int): The maximum number of recommendations to return.
        
        Returns:
            List[dict]: A list of recommended images with their IDs and scores.
        """

        recommended_images = client.recommend(
            collection_name=self.collection_name,
            positive=liked_images,
            negative=disliked_images,
            limit=limit
        )

        results = []
        for hit in recommended_images:
            results.append({
                "id": hit.id,
                "image_path": hit.payload.get("image_path"),
            })
        
        print(f"Recommended {len(results)} images based on {len(liked_images)} liked and {len(disliked_images)} disliked images.")
        print(f"Recommended images: {results}")
        return results

