from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import base64
from typing import Optional, List
from datetime import datetime
import os

from engine import generate_image

# Ensure the output directory exists
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI()

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
    png_files = sorted(
        [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".png")],
        key=lambda x: os.path.getmtime(os.path.join(OUTPUTS_DIR, x)),
        reverse=True
    )
    
    history = []
    for filename in png_files:
        prompt_path = os.path.join(OUTPUTS_DIR, os.path.splitext(filename)[0] + ".txt")
        prompt = ""
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r') as f:
                prompt = f.read()
        history.append({"filename": filename, "prompt": prompt})
        
    return {"history": history}

@app.delete("/api/history")
async def delete_all_history():
    """
    Deletes all files in the outputs directory.
    """
    for filename in os.listdir(OUTPUTS_DIR):
        file_path = os.path.join(OUTPUTS_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    return {"status": "History cleared"}

@app.delete("/api/history/item/{filename}") # Changed endpoint path
async def delete_history_item(filename: str):
    """
    Deletes a specific image and its corresponding prompt .txt file.
    """
    image_path = os.path.join(OUTPUTS_DIR, filename)
    prompt_path = os.path.join(OUTPUTS_DIR, os.path.splitext(filename)[0] + ".txt")
    
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        os.unlink(image_path)
        if os.path.exists(prompt_path):
            os.unlink(prompt_path)
        return {"status": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

@app.post("/generate")
async def generate_image_api(
    prompt: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    init_image: Optional[UploadFile] = File(None)
):
    init_image_bytes = await init_image.read() if init_image else None
    image = generate_image(prompt, width, height, init_image_bytes)
    
    # Save the image and the prompt to the outputs directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"img_{timestamp}"
    
    # Save image
    image_filename = f"{base_filename}.png"
    image_path = os.path.join(OUTPUTS_DIR, image_filename)
    image.save(image_path)

    # Save prompt
    try:
        prompt_filename = f"{base_filename}.txt"
        prompt_path = os.path.join(OUTPUTS_DIR, prompt_filename)
        with open(prompt_path, 'w') as f:
            f.write(prompt)
    except Exception as e:
        print(f"!!! ERROR: Could not save prompt to {prompt_path}. Error: {e}") # Added for debugging
    
    # Convert image to base64 string for immediate display
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}
