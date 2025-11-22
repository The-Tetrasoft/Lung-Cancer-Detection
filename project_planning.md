Project Planning - LungDx Backend Migration

Goal:
Replace Supabase demo functions with a FastAPI backend using MongoDB and expose the ML model for X-ray analysis.

What I added:
- backend/app.py: FastAPI server with auth, static pages, analyze, and medical-history endpoints.
- backend/model/loader.py: Model wrapper (loads PyTorch model if placed at backend/model/model.pth; otherwise returns mock results).
- backend/requirements.txt and backend/README.md

Frontend mapping (what to update):
- POST /api/login  <- replace previous supabase login fetch
- POST /api/signup <- replace previous supabase signup fetch
- POST /api/check-username <- replace previous supabase check
- GET /api/static/{page} <- replace previous static fetch
- POST /api/analyze <- new endpoint for real model inference

Data storage:
- MongoDB collections: users, medical_history

Next actions recommended:
1. Update frontend base URL to backend (I will update `src/utils/supabase/info.tsx`).
2. Install backend deps and run server locally.
3. Place trained PyTorch model at backend/model/model.pth to enable real inference.
