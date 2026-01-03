from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
import base64
from typing import Optional # Import Optional

from engine import load_model, generate_image

app = FastAPI()

# Load the model when the FastAPI application starts
@app.on_event("startup")
async def startup_event():
    load_model()

class GenerateRequest(BaseModel):
    prompt: str
    width: Optional[int] = None # Add width, make it optional
    height: Optional[int] = None # Add height, make it optional

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_image_api(request: GenerateRequest):
    image = generate_image(request.prompt, request.width, request.height) # Pass width and height
    
    # Convert image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": img_str}
