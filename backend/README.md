# LungDx Backend (FastAPI)

This backend provides REST endpoints for the LungDx frontend. It uses FastAPI, MongoDB (motor), and a model wrapper that will load a PyTorch model if provided at `backend/model/model.pth`.

Quick start (local):

1. Create a virtual environment and install dependencies:

```pwsh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run MongoDB locally (or set `MONGO_URI`).

3. Start the server:

```pwsh
#$env:PORT=8000
uvicorn app:app --reload --port 8000
```

Endpoints:
- POST /api/login
- POST /api/signup
- POST /api/check-username
- GET /api/static/{page}
- POST /api/analyze (multipart image)
- GET /api/medical-history/{username}

Place your trained PyTorch model at `backend/model/model.pth` if you want real predictions. The server will fall back to a deterministic mock when no model is present.