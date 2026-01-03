# LocalGen AI

This project is a monorepo for LocalGen AI, a self-hosted image generator. It follows a microservices strategy, allowing for independent development and deployment of its components.

## Project Structure

- `generation-service/`: The backend service responsible for AI image generation.
- `frontend-client/`: The web-based user interface for interacting with the generation service.

## Backend: Generation Service

The `generation-service` is built with Python 3.10+ and FastAPI. It leverages `diffusers` (Hugging Face) and `torch` (PyTorch) for machine learning, specifically using Stable Diffusion v1.5 or SDXL Turbo models.

**Capabilities:**
- Text-to-Image (txt2img)
- Image-to-Image (img2img)

## Frontend: Client

The `frontend-client` is a lightweight web application built with Vanilla HTML5, CSS3, and TypeScript. Vite is used for bundling during development. It communicates with the backend using the Fetch API.

## Future Enhancements

Future plans include the integration of a NestJS API Gateway for robust authentication and authorization.