from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from gradio_imageslider import ImageSlider
from loguru import logger

from func import DepthAnythingPipeline, string_to_port_crc32

# Constants
LOCAL_CLIENT_IP: str = "0.0.0.0"
APP_NAME: str = "depth_estimator"
# port number: 45944
DEFAULT_PORT: int = string_to_port_crc32(APP_NAME)


def setup_gradio_interface() -> gr.Blocks:
    """
    Sets up the Gradio interface for the depth estimation application.

    :return: Configured Gradio Blocks interface.
    """

    with gr.Blocks(title="DepthEstimator") as interface:
        with gr.Row():
            gr.Markdown(
                "# [Depth Estimator](https://github.com/CyFeng16/depth_estimator)"
            )
        with gr.Row():
            input_image = gr.Image(
                label="Input Image", type="filepath", elem_id="img-display-input"
            )
            depth_image_slider = ImageSlider(
                label="Depth Map with Slider View",
                elem_id="img-display-output",
                position=0.5,
            )
        with gr.Row():
            est_btn = gr.Button("Estimate!")
        est_btn.click(
            fn=estimate_depth_wrapper,
            inputs=[input_image],
            outputs=[depth_image_slider],
        )
    return interface


def estimate_depth(image_path: Path) -> np.ndarray:
    """
    Estimates the depth map from an input image using DepthAnythingPipeline.

    :param image_path: The path to the input image.
    :return: The estimated depth map as a NumPy array.
    """
    pipeline = DepthAnythingPipeline()
    logger.info(f"Estimating depth for image: {image_path}")
    depth = np.array(pipeline(image_path)["depth"])
    return cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]


def estimate_depth_wrapper(image_fp: str) -> tuple:
    """
    Wraps the depth estimation process and handles errors.

    :param image_fp: File path of the image to process.
    :return: A tuple containing the original image and the estimated depth map.
    """
    try:
        return (image_fp, estimate_depth(Path(image_fp)))
    except Exception as e:
        return (image_fp, f"Error during estimation: {e}")


def model_init_wrapper():
    return estimate_depth("/workspace/ice-festival.webp")


if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.queue()
    demo.launch(server_name=LOCAL_CLIENT_IP, server_port=DEFAULT_PORT)
