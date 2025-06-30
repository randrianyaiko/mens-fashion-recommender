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
        Generate a list of recommended images based on user preferences.

        This method uses the provided lists of liked and disliked image IDs to fetch 
        similar images that the user may also like. It returns a limited number of 
        image recommendations, each with its ID and file path.

        Args:
            liked_images (List[str]): A list of image IDs that the user liked.
            disliked_images (List[str]): A list of image IDs that the user disliked.
            limit (int, optional): Maximum number of recommended images to return. Default is 10.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains:
                - 'id' (str): The ID of the recommended image.
                - 'image_path' (str): The file path of the recommended image.
        """

        recommended_images = client.recommend(collection_name=self.collection_name,
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
        
        return results

