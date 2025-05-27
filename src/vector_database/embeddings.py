from fastembed import ImageEmbedding
from typing import List

model = ImageEmbedding("Qdrant/resnet50-onnx")

def compute_image_embedding(image_paths: List[str]) -> list[float]:
    """
    Compute the embedding for an image.

    Args:
        image (List[str]): The image paths to compute the embedding for, represented as a list of strings.
    Returns:
        list[float]: The computed embedding as a list of floats.
    """
    return list(model.embed(image_paths))