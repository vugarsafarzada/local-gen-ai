from fastapi import FastAPI, Form, File, UploadFile
from io import BytesIO
import base64
from typing import Optional

from engine import load_model, generate_image

app = FastAPI()

# Load the model when the FastAPI application starts
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_image_api(
    prompt: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    init_image: Optional[UploadFile] = File(None)
):
    init_image_bytes = await init_image.read() if init_image else None
    image = generate_image(prompt, width, height, init_image_bytes)
    
    # Convert image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}
