from qdrant_client import QdrantClient, models
from src.vector_database.embeddings import compute_image_embedding
import uuid, os, logging
from typing import List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
logger = logging.getLogger(__name__)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    
)
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

class VectorStore:
    def __init__(self, embed_batch: int = 64, upload_batch: int = 32, parallel_uploads: int = 3):
        """
        Initialize a VectorStore object.

        Args:
            embed_batch (int, optional): How many images to embed at once. Defaults to 64.
            upload_batch (int, optional): How many points to upload at once. Defaults to 32.
            parallel_uploads (int, optional): How many threads to use for uploading. Defaults to 3.

        Raises:
            ValueError: If the Qdrant collection does not exist, it is created.
        """
        self.client = client
        self.collection_name = collection_name
        self.embed_batch = embed_batch
        self.upload_batch = upload_batch
        self.parallel_uploads = max(1, parallel_uploads - 1)

        if not self.client.collection_exists(self.collection_name):
            self.create_collection()
        self.count_points()
        self.get_points()

    def create_collection(self):
        """
        Create the Qdrant collection if it does not exist.

        This method creates a collection on Qdrant with the name specified in the environment variable
        QDRANT_COLLECTION_NAME. The collection is configured to store vectors of size 512 with a COSINE
        distance metric.

        Raises:
            QdrantException: If the collection already exists.
        """
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        logger.info(f"Created collection {self.collection_name}")

    def count_points(self):
        """
        Get the number of points in the collection.

        This method queries the Qdrant collection specified in the environment variable
        QDRANT_COLLECTION_NAME for the number of points it contains and stores the result
        in the instance variable n_points. If an exception occurs during the query, the
        n_points variable is set to 0 and the exception is logged.

        Raises:
            QdrantException: If the collection does not exist or if the query fails.
        """
        try:
            self.n_points = self.client.count(self.collection_name).count
        except Exception as e:
            logger.error("[count_points] %s", e)
            self.n_points = 0

    def get_points(self):
        """
        Retrieve all points in the collection.

        This method queries the Qdrant collection specified in the environment variable
        QDRANT_COLLECTION_NAME for all points it contains and stores the result in the instance
        variable points. If an exception occurs during the query, the points variable is set
        to an empty list and the exception is logged.

        Raises:
            QdrantException: If the collection does not exist or if the query fails.
        """
        if self.n_points == 0:
            self.points = []
            return
        try:
            pts, _ = self.client.scroll(collection_name=self.collection_name, limit=self.n_points)
            self.points = [{"id": p.id, "image_path": p.payload.get("image_path")} for p in pts]
        except Exception as e:
            logger.error("[get_points] %s", e)
            self.points = []

    def insert_images(self, image_paths: List[str]):
        """
        Insert multiple images into the Qdrant collection.

        This method takes a list of image paths, embeds them in batches, creates points for the embeddings,
        and uploads them to Qdrant in batches. After all uploads are complete, the method refreshes the
        instance's knowledge of the number of points in the collection and the points themselves.

        Args:
            image_paths (List[str]): A list of paths to the images to be inserted.
        """
        def chunked(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]

        # Step 1: For each embedding batch, compute embeddings, create points, and upload immediately
        for batch in chunked(image_paths, self.embed_batch):
            embeddings = compute_image_embedding(batch)  # Batch embed
            points = [
                models.PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"image_path": img})
                for emb, img in zip(embeddings, batch)
            ]

            # Batch upload each sub-batch
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
                batch_size=self.upload_batch,
                parallel=self.parallel_uploads,
                max_retries=3,
                wait=True
            )

        # Step 2: Refresh stats
        self.count_points()
        self.get_points()

    def search_similar(self, query_image_path: str, limit: int = 5):
        """
        Find the most similar images to a query image.

        This method takes the path to a query image, computes its embedding, and searches the
        collection for the most similar images. It returns a list of dictionaries, where each
        dictionary contains the ID and image path of a similar image.

        Args:
            query_image_path (str): The path to the query image.
            limit (int, optional): The maximum number of similar images to return. Defaults to 5.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains:
                - 'id' (str): The ID of the similar image.
                - 'image_path' (str): The path to the similar image.
        """
        emb_list = compute_image_embedding([query_image_path])
        if not emb_list:
            return []
        hits = self.client.search(collection_name=self.collection_name, query_vector=emb_list[0], limit=limit)
        return [{"id": h.id, "image_path": h.payload.get("image_path")} for h in hits]
