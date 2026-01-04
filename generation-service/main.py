from fastapi import FastAPI, Form, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import base64
from typing import Optional, List
from datetime import datetime
import os
import uuid
import asyncio
import threading
from pydantic import BaseModel

from engine import generate_image, load_model, get_available_models, get_current_model_name, get_available_loras
import storage

# Ensure the output directory exists
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI()

class SwitchModelRequest(BaseModel):
    model_name: str

@app.on_event("startup")
async def startup_event():
    """
    Load the models on startup.
    """
    print("--- Application startup: Loading default model...")
    load_model("Default")
    print("--- Models loaded successfully.")

# Mount the outputs directory to serve static files
app.mount(f"/{OUTPUTS_DIR}", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/models")
async def list_models():
    """
    Returns a list of available models.
    """
    return {"models": get_available_models()}

@app.get("/api/loras")
async def list_loras():
    """
    Returns a list of available LoRAs.
    """
    return {"loras": get_available_loras()}

@app.post("/api/models/switch")
async def switch_model(request: SwitchModelRequest):
    load_model(request.model_name)
    return {"status": "success", "model": request.model_name}

@app.get("/api/history")
async def get_history():
    """
    Returns a list of all generated images and their prompts, sorted by newest first.
    """
    # Load history from JSON and reverse it to show newest first
    history = storage.load_history()[::-1]
    return {"history": history}

@app.delete("/api/history")
async def delete_all_history():
    """
    Deletes all files in the outputs directory.
    """
    storage.clear_history()
    return {"status": "History cleared"}

@app.delete("/api/history/item/{item_id}")
async def delete_history_item(item_id: str):
    """
    Deletes a specific image by ID.
    """
    success = storage.delete_history_item(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"status": f"Deleted {item_id}"}

@app.post("/generate")
async def generate_image_api(
    prompt: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    negative_prompt: Optional[str] = Form(None),
    guidance_scale: float = Form(7.5),
    num_inference_steps: int = Form(30),
    lora: Optional[str] = Form(None),
    scheduler: str = Form("Euler a"), 
    use_face_swap: bool = Form(False), # New parameter
    init_image: Optional[UploadFile] = File(None)
):
    init_image_bytes = await init_image.read() if init_image else None
    image = generate_image(
        prompt=prompt,
        width=width,
        height=height,
        init_image_bytes=init_image_bytes,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        lora_name=lora,
        scheduler=scheduler,
        use_face_swap=use_face_swap # Pass to engine
    )
    
    # Generate UUID and Filename
    image_id = str(uuid.uuid4())
    image_filename = f"{image_id}.png"
    
    # Save image
    image_path = os.path.join(OUTPUTS_DIR, image_filename)
    image.save(image_path)

    # Save metadata to JSON
    metadata = {
        "id": image_id,
        "filename": image_filename,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "settings": {
            "width": width,
            "height": height,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "model": get_current_model_name(),
            "lora": lora,
            "scheduler": scheduler,
            "use_face_swap": use_face_swap,
            "seed": 0 # Placeholder
        }
    }
    storage.add_history_item(metadata)
    
    # Convert image to base64 string for immediate display
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stop_event = threading.Event()
    try:
        data = await websocket.receive_json()
        
        # Extract parameters
        prompt = data.get("prompt")
        width = int(data.get("width", 512))
        height = int(data.get("height", 512))
        negative_prompt = data.get("negative_prompt")
        guidance_scale = float(data.get("guidance_scale", 7.5))
        num_inference_steps = int(data.get("num_inference_steps", 30))
        init_image_b64 = data.get("init_image")
        lora = data.get("lora")
        scheduler = data.get("scheduler", "Euler a")
        use_face_swap = data.get("use_face_swap", False) # New parameter
        
        init_image_bytes = None
        if init_image_b64:
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if "," in init_image_b64:
                init_image_b64 = init_image_b64.split(",")[1]
            init_image_bytes = base64.b64decode(init_image_b64)

        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Define callback to send progress
        def progress_callback(step, timestep, latents):
            # Calculate percentage (step is 0-indexed)
            percentage = int(((step + 1) / num_inference_steps) * 100)
            
            if stop_event.is_set():
                raise RuntimeError("Generation cancelled")
            
            # print(f"Generation Progress: {percentage}%") # Optional: reduce log noise
            msg = {"status": "progress", "percentage": percentage, "step": step + 1, "total_steps": num_inference_steps}
            # Send message to websocket from the thread
            asyncio.run_coroutine_threadsafe(websocket.send_json(msg), loop)

        # Run generation in a separate thread to avoid blocking the websocket loop
        try:
            image = await loop.run_in_executor(None, lambda: generate_image(
                prompt=prompt, width=width, height=height, init_image_bytes=init_image_bytes, 
                negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps, lora_name=lora, scheduler=scheduler, 
                use_face_swap=use_face_swap, callback=progress_callback
            ))
        except asyncio.CancelledError:
            print("Generation task cancelled.")
            stop_event.set()
            raise

        # Save and process result (similar to POST endpoint)
        image_id = str(uuid.uuid4())
        image_filename = f"{image_id}.png"
        image_path = os.path.join(OUTPUTS_DIR, image_filename)
        image.save(image_path)

        metadata = {
            "id": image_id,
            "filename": image_filename,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "width": width, "height": height, "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale, "num_inference_steps": num_inference_steps,
                "model": get_current_model_name(), "lora": lora, "scheduler": scheduler, 
                "use_face_swap": use_face_swap, "seed": 0
            }
        }
        storage.add_history_item(metadata)

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        await websocket.send_json({"status": "completed", "image": img_str})

    except WebSocketDisconnect:
        print("Client disconnected")
        stop_event.set()
    except Exception as e:
        print(f"Error in websocket: {e}")
        # Only try to send error if socket is likely open (not cancelled/disconnected)
        if not stop_event.is_set():
            await websocket.send_json({"status": "error", "message": str(e)})