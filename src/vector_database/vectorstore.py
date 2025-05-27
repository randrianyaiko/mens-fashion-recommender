from qdrant_client import QdrantClient
from qdrant_client import models
from src.vector_database.embeddings import compute_image_embedding
import uuid
import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

class VectorStore:
    def __init__(self):
        self.client = client
        self.collection_name = collection_name
        self.countPoints()
        self.getPoints()

    def countPoints(self):
        try:
            self.n_points = self.client.count(collection_name=self.collection_name).count
        except Exception as e:
            print(f"Error counting points. So setting count to 0")
            self.n_points = 0
        
    def getPoints(self):
        try:
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=self.n_points)
            
            points = points[0]
            points = [{"id": point.id,"image_path": point.payload.get("image_path")} for point in points]
            self.points = points
        except Exception as e:
            print(f"Error retrieving points. So setting points to empty list")
            self.points = []

    def insertImagesToVDB(self, image_paths: List[str]):
        embeddings = compute_image_embedding(image_paths)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"image_path": image_path}
            )
            for embedding, image_path in zip(embeddings, image_paths)
        ]
        
        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=64,
        )

        self.countPoints()
        self.getPoints()
        return True

    def searchSimilarImages(self, query_image_path: str, limit: int = 5):
        query_embedding = compute_image_embedding([query_image_path])[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for hit in search_result:
            results.append({
                "id": hit.id,
                "image_path": hit.payload.get("image_path")
            })
        
        return results
