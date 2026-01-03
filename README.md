# LocalGen AI

LocalGen AI is a self-hosted, privacy-first AI image generation platform. It runs entirely locally on your hardware, ensuring no external API keys or cloud dependencies are required.

## (+) Features

*   **Text-to-Image:** Generate high-quality images from text prompts.
*   **Custom Model Support:** Switch between different Stable Diffusion checkpoints (`.safetensors`, `.ckpt`) stored locally.
*   **SDXL Ready:** Seamlessly supports both Stable Diffusion v1.5 and SDXL architectures.
*   **Image-to-Image:** Use an initial image as a base for your generations.
*   **Advanced Controls:** Fine-tune your results with:
    *   Negative Prompts
    *   Guidance Scale
    *   Inference Steps
    *   Custom Dimensions (Width/Height)
*   **Real-time Feedback:** Live progress bar powered by WebSockets.
*   **History Gallery:** Automatically saves generated images and metadata (including model used). Click history items to restore settings.
*   **Privacy Focused:** All processing happens on your machine.

## (#) Technical Stack

### Backend (Generation Service)
*   **Framework:** FastAPI (Python 3.10+)
*   **ML Engine:** Hugging Face `diffusers`, PyTorch
*   **Storage:** JSON-based metadata & local filesystem

### Frontend (Client)
*   **Core:** HTML5, CSS3, TypeScript
*   **Build Tool:** Vite
*   **Communication:** WebSockets & Fetch API

## [-] Project Structure

This project follows a monorepo structure:

```text
/
├── apps/
│   ├── generation-service/    # Python/FastAPI Backend
│   │   ├── main.py            # API & WebSocket Entry point
│   │   ├── engine.py          # ML Inference Logic
│   │   └── storage.py         # JSON History Management
│   │
│   └── frontend-client/       # TypeScript Frontend
│       ├── src/               # UI Logic & Styles
│       └── index.html         # Entry HTML
```

## [-] Quick Start

### Prerequisites
*   **Python:** 3.10 or higher
*   **Node.js:** 18 or higher
*   **Hardware:** NVIDIA GPU (CUDA) or Apple Silicon (MPS) recommended.

### 1. Start the Backend
```bash
cd apps/generation-service
pip install -r requirements.txt
python main.py
```

### 2. Start the Frontend
```bash
cd apps/frontend-client
npm install
npm run dev
```

### 3. Adding Custom Models

1.  Create a directory named `models` inside `apps/generation-service/` (if it doesn't exist).
2.  Download Stable Diffusion checkpoints (e.g., from Civitai or Hugging Face).
3.  Place `.safetensors` or `.ckpt` files into the `models/` directory.
4.  Refresh the web interface; the new models will appear in the "Model" dropdown.

## [-] Future Enhancements

*   Integration of a NestJS API Gateway for robust authentication and authorization.