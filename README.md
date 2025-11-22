
# LungDx — Lung Cancer Detection App

LungDx is a full-stack web application for AI-assisted analysis of chest X-rays. The project includes a React + Vite frontend, a FastAPI backend that exposes REST endpoints for authentication, medical-history storage, and image analysis, and a small model loader that can run a TensorFlow Lite or PyTorch model (or a deterministic fallback). This repo is a developer bundle intended to run locally for experimentation and demonstration.

## Key Features

- React + Vite frontend with user and admin views
- FastAPI backend with endpoints for signup, login, analysis, and medical history
- MongoDB persistence (via `motor` async driver)
- Model inference via TFLite or PyTorch (with a provided `densenet169_best.tflite` model)
- Simple user roles (user / admin) and basic account locking on failed login

## Tech Stack

- Frontend: React + TypeScript, Vite
- Backend: Python, FastAPI, Uvicorn
- Database: MongoDB (async via `motor`)
- ML runtime: TensorFlow/TFLite or PyTorch (optional), with tflite-runtime as an alternative

## Repository Structure (important files)

- `src/` — Frontend React app (Vite) and components
- `backend/` — FastAPI application and backend code
  - `backend/app.py` — main FastAPI app and endpoints
  - `backend/model/loader.py` — ModelWrapper supporting TFLite or torch
- `model/` — contains `densenet169_best.tflite` (TFLite model included)
- `package.json` — frontend dependencies and scripts
- `backend/requirements.txt` — Python dependencies for backend

## Prerequisites

- Node.js (16+) and npm/yarn for frontend
- Python 3.10+ (recommended) for backend
- MongoDB instance (local or remote)

Optional for model acceleration:
- `tensorflow` or `tflite-runtime` (for TFLite interpreter)
- `torch` (if you want PyTorch model support)

## Local Development — Frontend

1. Install frontend dependencies:

```bash
npm install
```

2. Run the dev server (Vite):

```bash
npm run dev
```

The frontend will typically be available at `http://localhost:5173` (Vite prints the exact URL).

## Local Development — Backend

1. Create and activate a Python virtual environment:

```bash
# create venv
python -m venv .venv

# activate (Bash/Git Bash)
source .venv/Scripts/activate

# if you're on WSL/macOS/Linux use:
# source .venv/bin/activate
```

2. Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Environment variables (optional):

- `MONGO_URI` — MongoDB connection string (default: `mongodb://localhost:27017`)
- `PORT` — backend port used when running via `python` entry-point (default: `8000`)

You can export them in Bash:

```bash
export MONGO_URI="mongodb://localhost:27017"
export PORT=8000
```

4. Start the backend using Uvicorn from the repo root:

```bash
# run from repo root
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## Model details

- A TFLite model is included at `model/densenet169_best.tflite`. The backend's `backend/model/loader.py` attempts to load a TFLite interpreter first (via `tensorflow` or `tflite_runtime`), then looks for a PyTorch model at `backend/model/model.pth`.
- If no ML runtime is available, the loader falls back to a deterministic hash-based mock prediction for demo purposes.

Notes on installing TensorFlow/TFLite on Windows:
- Installing full `tensorflow` on Windows can be heavy and sometimes problematic. If you only need to run the included TFLite model, prefer installing `tflite-runtime` for your platform (prebuilt wheel), or run the backend in Linux/Docker where `tensorflow` is easier to install.

## API examples

- Signup

POST /api/signup — JSON body per `SignupRequest` fields.

- Login

POST /api/login — JSON body with `usernameOrEmail` and `password`.

- Analyze image (multipart form)

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@/path/to/chest_xray.jpg" \
  -F "username=testuser"
```

Response (example):

```json
{
  "prediction": "Negative",
  "confidence": 12.3,
  "riskLevel": "Low",
  "detailedMetrics": { /* ... */ },
  "recommendations": ["Routine monitoring"]
}
```

## MongoDB / Data

- The backend uses a database named `lungdx` by default and stores users and `medical_history` entries. You can change the `MONGO_URI` to point to a hosted MongoDB Atlas cluster.

## Deployment notes

- For production, consider:
  - Running the backend behind a production ASGI server (Gunicorn + Uvicorn workers) or containerizing with Docker.
  - Using a managed MongoDB instance.
  - Serving the frontend as a static build (Vite build) behind a CDN or static host.
  - Securing secrets and enabling HTTPS.

## Contributing

If you'd like to contribute, open an issue first to describe the change. Follow these steps for code contributions:

1. Fork the repository
2. Create a feature branch
3. Add tests where applicable
4. Submit a pull request with a clear description

## License

No license file is included in this repository. Add a `LICENSE` file (for example MIT) if you want to make the project open-source.

  
