import zlib
from pathlib import Path
from typing import Union

import torch
from loguru import logger
from transformers import Pipeline, pipeline

# Define constants for clarity and easy maintenance.
MIN_PORT: int = 10000
MAX_PORT: int = 60000
DEFAULT_MODEL_ID: str = "LiheYoung/depth-anything-large-hf"


def string_to_port_crc32(input_string: str) -> int:
    """
    Converts a string to a CRC32 hash and then maps it to a port number within a specified range.
    """

    crc32_hash: int = zlib.crc32(input_string.encode()) & 0xFFFFFFFF
    port_number: int = MIN_PORT + crc32_hash % (MAX_PORT - MIN_PORT)
    return port_number


def set_device() -> torch.device:
    """
    Determines the best available device for PyTorch operations.

    :return: The optimal torch.device instance.
    """

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    return device


class DepthAnythingPipeline:
    """
    A class for running the depth estimation pipeline on an image file.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        self.model_id: str = model_id
        self.device: torch.device = set_device()

    @staticmethod
    def validate_image_path(image_path: Union[str, Path]) -> Path:
        """
        Validates and converts the image path to a Path object.

        :param image_path: The image path to validate.
        :return: The validated Path object.
        :raises FileNotFoundError: If the image file does not exist.
        """

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file {path} not found.")
        return path

    def __call__(self, image_path: Union[str, Path]):
        """
        Executes the depth estimation pipeline on the provided image path.

        :param image_path: The path to the image file.
        :return: The result from the depth estimation pipeline.
        """

        path = self.validate_image_path(image_path)
        depth_pipeline = self.setup_pipeline()
        return depth_pipeline(str(path))

    def setup_pipeline(self) -> Pipeline:
        """
        Sets up the depth estimation pipeline.

        :return: The initialized pipeline.
        """

        try:
            depth_pipeline = pipeline(
                task="depth-estimation", model=self.model_id, device=self.device
            )
            logger.info("Model loaded successfully.")
            return depth_pipeline
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
