from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch
from typing import Optional
from PIL import Image
from io import BytesIO

# We'll manage two separate pipelines now
txt2img_pipeline = None
img2img_pipeline = None

def load_txt2img_model():
    """
    Loads the Stable Diffusion Text-to-Image model.
    """
    global txt2img_pipeline
    if txt2img_pipeline is None:
        print("--- Loading Text-to-Image model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        txt2img_pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        txt2img_pipeline.to("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return txt2img_pipeline

def load_img2img_model():
    """
    Loads the Stable Diffusion Image-to-Image model.
    """
    global img2img_pipeline
    if img2img_pipeline is None:
        print("--- Loading Image-to-Image model...")
        model_id = "runwayml/stable-diffusion-v1-5" # Can use the same base model
        img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        img2img_pipeline.to("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return img2img_pipeline

def generate_image(
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    init_image_bytes: Optional[bytes] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30
):
    """
    Generates an image from a text prompt, optionally using an initial image.
    """
    # Set default dimensions if not provided
    width = width or 512
    height = height or 512

    if init_image_bytes:
        # Image-to-Image generation
        pipeline = load_img2img_model()
        init_image = Image.open(BytesIO(init_image_bytes)).convert("RGB")
        init_image = init_image.resize((width, height))
        
        # Simple generation for now, can be expanded with more parameters later
        image = pipeline(
            prompt=prompt, 
            image=init_image, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps
        ).images[0]

    else:
        # Text-to-Image generation
        pipeline = load_txt2img_model()
        image = pipeline(
            prompt=prompt, 
            width=width, 
            height=height, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps
        ).images[0]
        
    return image