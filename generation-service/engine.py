from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionPipeline, StableDiffusionXLPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler, UniPCMultistepScheduler
import torch
from typing import Optional
from PIL import Image
from io import BytesIO
import os
import gc
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Global pipeline management
current_pipeline = None
current_model_name = "Default"
MODELS_DIR = "models"
LORA_DIR = "loras"

# Global Face Analysis & Swapper
face_app = None
face_swapper = None

def init_face_modules():
    global face_app, face_swapper
    
    # Initialize Face Analysis (Detection)
    # allow_face_prob ensures we only get high-quality faces
    face_app = FaceAnalysis(name='buffalo_l')
    # Use context_dir to avoid downloading to home directory if possible, or just let it handle defaults.
    # For this environment, we rely on default download or pre-downloaded models if 'buffalo_l' is standard.
    # If internet is restricted, this might fail if buffalo_l isn't cached.
    # However, user only mentioned 'inswapper_128.onnx' being present. 
    # FaceAnalysis usually downloads its own detection models. 
    # We will assume it can download or they are present.
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize Swapper
    swapper_path = os.path.join(MODELS_DIR, "insightface", "inswapper_128.onnx")
    if os.path.exists(swapper_path):
        face_swapper = insightface.model_zoo.get_model(swapper_path, download=False, download_zip=False)
    else:
        print(f"Warning: Face swapper model not found at {swapper_path}")

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

    # Common kwargs to disable safety checker for all pipelines
    pipeline_kwargs = {
        "safety_checker": None,
        "requires_safety_checker": False
    }

    if model_name == "Default":
        model_id = "runwayml/stable-diffusion-v1-5"
        current_pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, **pipeline_kwargs)
    else:
        model_path = os.path.join(MODELS_DIR, model_name)
        print(f"--- Loading model: {model_name}")

        # 1. FLUX Check (Specific filename check)
        if "flux" in model_name.lower():
            print("--- Detected Strategy: FLUX")
            try:
                current_pipeline = FluxPipeline.from_single_file(
                    model_path, 
                    torch_dtype=torch.bfloat16, 
                    use_safetensors=True, 
                    local_files_only=True
                )
                current_pipeline.enable_model_cpu_offload()
                current_pipeline.to(device)
                current_model_name = model_name
                print(f"--- Model {model_name} loaded successfully (Flux).")
                return current_pipeline
            except Exception as e:
                print(f"Flux load failed: {e}")
                raise e

        # 2. SDXL Strategy (Generic #1 - Priority)
        print("--- Strategy 1: Attempting load as SDXL...")
        # Smart Heuristic: Check for 'xl' OR 'pony' to prefer SDXL pipeline
        if any(x in model_name.lower() for x in ['xl', 'pony']):
             print(f"--- Heuristic: '{model_name}' detected as likely SDXL/Pony.")

        try:
            current_pipeline = StableDiffusionXLPipeline.from_single_file(
                model_path, 
                torch_dtype=torch.float16, 
                use_safetensors=True,
                **pipeline_kwargs
            )
            print("--- Success: Loaded as SDXL.")
        except Exception as e_xl:
            print(f"--- SDXL load failed. Switching to Strategy 2 (SD3). Error: {str(e_xl)[:100]}...")
            
            # 3. SD3 Strategy (Fallback for newer models like Qwen/T5 based)
            print("--- Strategy 2: Attempting load as SD3...")
            try:
                current_pipeline = StableDiffusion3Pipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    **pipeline_kwargs
                )
                print("--- Success: Loaded as SD3.")
            except Exception as e_sd3:
                print(f"--- SD3 load failed. Switching to Strategy 3 (Flux - Generic). Error: {str(e_sd3)[:100]}...")

                # 4. Flux Strategy (Generic Fallback)
                print("--- Strategy 3: Attempting load as Flux (Generic)...")
                try:
                    current_pipeline = FluxPipeline.from_single_file(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        use_safetensors=True,
                        local_files_only=True
                    )
                    current_pipeline.enable_model_cpu_offload()
                    print("--- Success: Loaded as Flux (Generic).")
                except Exception as e_flux:
                    print(f"--- Flux load failed. Switching to Strategy 4 (SD1.5). Error: {str(e_flux)[:100]}...")

                    # 5. SD1.5 Strategy (Final Fallback)
                    print("--- Strategy 4: Attempting load as SD1.5 (Standard)...")
                    try:
                        current_pipeline = StableDiffusionPipeline.from_single_file(
                            model_path, 
                            torch_dtype=torch.float16, 
                            use_safetensors=True,
                            **pipeline_kwargs
                        )
                        print("--- Success: Loaded as SD1.5.")
                    except Exception as e_sd:
                        # If all fail, raise a clear error
                        raise ValueError(f"Model architecture not recognized.\nSDXL: {e_xl}\nSD3: {e_sd3}\nFlux: {e_flux}\nSD1.5: {e_sd}\n\nHint: This model might be 'Text-Encoder-Free' or requires newer libraries.")
    
    current_pipeline.to(device)
    current_model_name = model_name
    print(f"--- Model {model_name} loaded successfully.")
    return current_pipeline

def set_scheduler(pipeline, scheduler_id: str):
    """
    Sets the scheduler for the given pipeline based on the scheduler_id.
    """
    if scheduler_id == "Euler a":
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_id == "Euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_id == "DPM++ 2M Karras":
        # For DPM++ 2M Karras, specific config is often recommended
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++"
        )
    elif scheduler_id == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_id == "UniPC":
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    else: # Default to Euler a
        print(f"--- Warning: Unknown scheduler '{scheduler_id}'. Defaulting to Euler a.")
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

def generate_image(
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    init_image_bytes: Optional[bytes] = None,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    lora_name: Optional[str] = None,
    scheduler: str = "Euler a",
    use_face_swap: bool = False,
    callback = None
):
    """
    Generates an image from a text prompt, optionally using an initial image and applying a LoRA.
    """
    # Set default dimensions if not provided
    width = width or 512
    height = height or 512

    # Input Sanitization
    if negative_prompt is None:
        negative_prompt = ""
    if lora_name is None or lora_name == "None":
        lora_name = ""
    if prompt is None:
        prompt = ""

    global current_pipeline, current_lora_name, face_app, face_swapper
    if current_pipeline is None:
        load_model("Default")

    # Initialize Face Modules if needed and if face swap is requested
    if use_face_swap and (face_app is None or face_swapper is None):
        print("--- Initializing Face Swap modules...")
        init_face_modules()

    # Set the scheduler
    set_scheduler(current_pipeline, scheduler)

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

    image = None

    # Adapter for new callback style (Diffusers v0.23+)
    def progress_wrapper(pipe, step_index, timestep, callback_kwargs):
        if callback:
             # The old callback expected (step, timestep, latents)
             # We try to get latents from kwargs, though often not needed for simple progress bars
             latents = callback_kwargs.get("latents", None)
             callback(step_index, timestep, latents)
        return callback_kwargs

    if init_image_bytes and not use_face_swap:
        # Standard Image-to-Image generation (only if NOT in face swap mode)
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
            callback_on_step_end=progress_wrapper,
        ).images[0]

    else:
        # Text-to-Image generation (Used for standard Txt2Img AND Face Swap base)
        image = current_pipeline(
            prompt=prompt, 
            width=width, 
            height=height, 
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale, 
            num_inference_steps=num_inference_steps,
            callback_on_step_end=progress_wrapper,
        ).images[0]
    
    # Handle Face Swap Post-Processing
    if use_face_swap and init_image_bytes:
        print("--- Performing Face Swap...")
        if face_app is None or face_swapper is None:
             print("Error: Face modules not initialized. Skipping swap.")
        else:
            try:
                # 1. Prepare Source Face (from init_image_bytes)
                # Convert bytes to numpy array for cv2
                nparr = np.frombuffer(init_image_bytes, np.uint8)
                source_img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Detect faces in source
                source_faces = face_app.get(source_img_cv)
                if len(source_faces) == 0:
                    print("No face detected in source image.")
                else:
                    # Sort by size to get the largest face (main subject)
                    source_faces.sort(key=lambda x: x.bbox[2] * x.bbox[3], reverse=True)
                    source_face = source_faces[0]

                    # 2. Prepare Target Image (the generated image)
                    # Convert PIL to CV2
                    target_img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Detect faces in target
                    target_faces = face_app.get(target_img_cv)
                    if len(target_faces) == 0:
                        print("No face detected in generated image to swap.")
                    else:
                        # Swap faces
                        # We swap onto ALL faces detected in the target for now, or just the largest.
                        # Usually swapping onto all is fun, but swapping onto the largest/most prominent is safer.
                        # Let's swap onto all faces found in the target to ensure the character gets the face.
                        res_img = target_img_cv.copy()
                        for target_face in target_faces:
                            res_img = face_swapper.get(res_img, target_face, source_face, paste_back=True)
                        
                        # Convert back to PIL
                        image = Image.fromarray(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
                        print("--- Face Swap completed.")

            except Exception as e:
                print(f"Error during face swap: {e}")
        
    return image