from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import base64
from typing import Optional, List
from datetime import datetime
import os
import uuid

from engine import generate_image, load_txt2img_model, load_img2img_model
import storage

# Ensure the output directory exists
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Load the models on startup.
    """
    print("--- Application startup: Loading models...")
    load_txt2img_model()
    load_img2img_model()
    print("--- Models loaded successfully.")

# Mount the outputs directory to serve static files
app.mount(f"/{OUTPUTS_DIR}", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
def read_root():
    return {"Hello": "World"}

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
    init_image: Optional[UploadFile] = File(None)
):
    init_image_bytes = await init_image.read() if init_image else None
    image = generate_image(prompt, width, height, init_image_bytes, negative_prompt, guidance_scale, num_inference_steps)
    
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
            "model": "stable-diffusion-v1-5",
            "seed": 0 # Placeholder
        }
    }
    storage.add_history_item(metadata)
    
    # Convert image to base64 string for immediate display
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}
