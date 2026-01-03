from diffusers import DiffusionPipeline
import torch
from typing import Optional

pipeline = None

def load_model():
    """
    Loads the Stable Diffusion model.
    """
    global pipeline
    if pipeline is None:
        # Using a smaller, faster model for initial setup. Can be replaced with SDXL-Turbo later.
        model_id = "runwayml/stable-diffusion-v1-5"
        pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipeline.to("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return pipeline

def generate_image(prompt: str, width: Optional[int] = None, height: Optional[int] = None):
    """
    Generates an image from a text prompt using the loaded Stable Diffusion model.
    """
    if pipeline is None:
        load_model()
    
    # Set default dimensions if not provided
    if width is None:
        width = 512
    if height is None:
        height = 512

    # Generate image with specified width and height
    image = pipeline(prompt, width=width, height=height).images[0]
    return image