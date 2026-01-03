from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from typing import Optional
from PIL import Image
from io import BytesIO
import os
import gc

# Global pipeline management
current_pipeline = None
current_model_name = "Default"
MODELS_DIR = "models"
LORA_DIR = "loras"

def get_available_models():
    """
    Scans the models directory and returns a list of available models,
    filtering out files smaller than 1GB which are likely LoRAs.
    """
    models = ["Default"]
    if os.path.exists(MODELS_DIR):
        for file in os.listdir(MODELS_DIR):
            file_path = os.path.join(MODELS_DIR, file)
            if (file.endswith(".safetensors") or file.endswith(".ckpt")) and os.path.getsize(file_path) >= 1_000_000_000: # 1GB
                models.append(file)
    return models

def get_available_loras():
    """
    Scans the loras directory and returns a list of available LoRAs,
    including a "None" option.
    """
    loras = ["None"]
    if os.path.exists(LORA_DIR):
        for file in os.listdir(LORA_DIR):
            if file.endswith(".safetensors") or file.endswith(".ckpt"):
                loras.append(file)
    return loras

current_lora_name = None

def get_current_model_name():
    return current_model_name

def load_model(model_name: str):
    """
    Loads a specific model by name, handling memory cleanup.
    """
    global current_pipeline, current_model_name
    
    print(f"--- Switching to model: {model_name}...")
    
    # Memory Cleanup
    if current_pipeline is not None:
        del current_pipeline
        current_pipeline = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "Default":
        model_id = "runwayml/stable-diffusion-v1-5"
        current_pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
    else:
        model_path = os.path.join(MODELS_DIR, model_name)
        # Heuristic: Check if filename contains 'xl' to use SDXL pipeline, otherwise default to SD1.5
        if "xl" in model_name.lower():
            print(f"--- Detected SDXL model from filename: {model_name}")
            current_pipeline = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16)
        else:
            print(f"--- Detected SD1.5 model from filename: {model_name}")
            current_pipeline = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16)
    
    current_pipeline.to(device)
    current_model_name = model_name
    print(f"--- Model {model_name} loaded successfully.")
    return current_pipeline

def generate_image(
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    init_image_bytes: Optional[bytes] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    lora_name: Optional[str] = None, # New parameter
    callback = None
):
    """
    Generates an image from a text prompt, optionally using an initial image and applying a LoRA.
    """
    # Set default dimensions if not provided
    width = width or 512
    height = height or 512

    global current_pipeline, current_lora_name
    if current_pipeline is None:
        load_model("Default")

    # LoRA handling
    if lora_name != current_lora_name:
        if current_lora_name is not None and current_lora_name != "None":
            print(f"--- Unloading LoRA: {current_lora_name}")
            # Ensure the pipeline method for unloading LoRA exists before calling
            if hasattr(current_pipeline, 'unload_lora_weights'):
                current_pipeline.unload_lora_weights()
            else:
                print("Warning: unload_lora_weights not found on pipeline. May not be able to unload LoRA explicitly.")
            
        if lora_name is not None and lora_name != "None":
            lora_path = os.path.join(LORA_DIR, lora_name)
            print(f"--- Loading LoRA: {lora_name} from {lora_path}")
            current_pipeline.load_lora_weights(lora_path)
            current_lora_name = lora_name
        else:
            current_lora_name = None # No LoRA selected or "None" selected

    if init_image_bytes:
        # Image-to-Image generation
        init_image = Image.open(BytesIO(init_image_bytes)).convert("RGB")
        init_image = init_image.resize((width, height))
        
        # Create img2img pipeline from the current components to save memory
        img2img_pipe = AutoPipelineForImage2Image.from_pipe(current_pipeline)
        image = img2img_pipe(
            prompt=prompt,
            image=init_image, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps,
            callback=callback,
            callback_steps=1
        ).images[0]

    else:
        # Text-to-Image generation
        image = current_pipeline(
            prompt=prompt, 
            width=width, 
            height=height, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps,
            callback=callback,
            callback_steps=1
        ).images[0]
        
    return image