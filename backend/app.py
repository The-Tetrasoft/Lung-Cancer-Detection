from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import motor.motor_asyncio
import os
import base64
from datetime import datetime
import numpy as np
from PIL import Image

from io import BytesIO

import importlib

# Robust import for ModelWrapper: try several module paths depending on how uvicorn is started
ModelWrapper = None
for _mod in ("backend.model.loader", "model.loader", "model.loader"):
    try:
        _m = importlib.import_module(_mod)
        ModelWrapper = getattr(_m, 'ModelWrapper')
        break
    except Exception:
        continue

if ModelWrapper is None:
    raise ImportError("Could not import ModelWrapper from backend.model.loader or model.loader")

app = FastAPI(title="LungDx Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB setup - read MONGO_URI from env or use local default
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client['lungdx']

# DB health flag
db_ready = False

# Startup ping to ensure Mongo is reachable (non-fatal)
@app.on_event("startup")
async def startup_event():
    global db_ready
    try:
        # ping the server
        await client.admin.command('ping')
        db_ready = True
        print("MongoDB ping successful")
    except Exception as e:
        db_ready = False
        print("MongoDB ping failed:", e)

# Load model
model = ModelWrapper()


class LoginRequest(BaseModel):
    usernameOrEmail: str
    password: str


class SignupRequest(BaseModel):
    username: str
    firstName: str
    lastName: str
    dateOfBirth: str
    gender: str
    email: EmailStr
    phoneNumber: str
    password: str
    profilePicture: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[EmailStr] = None
    phoneNumber: Optional[str] = None
    dateOfBirth: Optional[str] = None
    gender: Optional[str] = None
    profilePicture: Optional[str] = None


@app.post("/api/check-username")
async def check_username(payload: dict):
    username = payload.get('username')
    if not username:
        raise HTTPException(status_code=400, detail='username required')
    existing = await db.users.find_one({"username": username})
    return {"available": existing is None}


@app.post("/api/signup")
async def signup(req: SignupRequest):
    existing = await db.users.find_one({"username": req.username})
    if existing:
        return JSONResponse({"success": False, "message": "Username exists"})
    
    existing_email = await db.users.find_one({"email": req.email})
    if existing_email:
        return JSONResponse({"success": False, "message": "Email already registered"})

    user_doc = req.dict()
    user_doc['password'] = base64.b64encode(req.password.encode('utf-8')).decode('utf-8')
    user_doc['role'] = 'user'
    user_doc['created_at'] = datetime.utcnow()
    user_doc['loginAttempts'] = 0
    await db.users.insert_one(user_doc)
    return {"success": True}


@app.post("/api/login")
async def login(payload: LoginRequest):
    q = {"$or": [{"username": payload.usernameOrEmail}, {"email": payload.usernameOrEmail}]}
    user = await db.users.find_one(q)
    if not user:
        return {"success": False, "message": "User not found", "remainingAttempts": 5}

    if user.get('loginAttempts', 0) >= 5:
        return JSONResponse({"success": False, "message": "Account locked due to too many failed attempts"}, status_code=423)

    decoded = base64.b64decode(user['password']).decode('utf-8')
    if decoded != payload.password:
        await db.users.update_one({"_id": user['_id']}, {"$inc": {"loginAttempts": 1}})
        remaining = max(0, 5 - (user.get('loginAttempts', 0) + 1))
        return {"success": False, "message": "Invalid credentials", "remainingAttempts": remaining}

    # success
    await db.users.update_one({"_id": user['_id']}, {"$set": {"loginAttempts": 0}})
    user['password'] = None
    return {"success": True, "user": {"username": user['username'], "firstName": user.get('firstName',''), "lastName": user.get('lastName',''), "email": user.get('email',''), "role": user.get('role','user')}}


@app.get("/api/static/{page}")
async def static_page(page: str):
    pages = {
        'aboutUs': {
            'title': 'About LungDx',
            'data': {
                'title': 'About LungDx',
                'content': 'We provide AI-assisted lung X-ray analysis.',
                'images': []
            }
        },
        'contact': {'title': 'Contact', 'data': {'title': 'Contact', 'content': 'Contact us at support@lungdx.example', 'images': []}},
        'terms': {'title': 'Terms', 'data': {'title': 'Terms', 'content': 'Terms of service...', 'images': []}}
    }
    p = pages.get(page)
    if not p:
        return {"success": False}
    return {"success": True, "data": p['data']}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), username: Optional[str] = Form(None)):
    # Read image bytes
    contents = await image.read()
    try:
        pil = Image.open(BytesIO(contents)).convert('RGB')
    except Exception:
        # Try to decode base64 inside payload if sent as text
        raise HTTPException(status_code=400, detail='Invalid image')

    # Run model prediction
    result = model.predict_image_bytes(contents)

    # Persist result to medical_history collection only if explicitly enabled
    if os.environ.get('PY_SAVE_TO_DB', '0') == '1':
        try:
            entry = {
                'username': username or 'anonymous',
                'date': datetime.utcnow(),
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'riskLevel': result['riskLevel'],
                'detailedMetrics': result.get('detailedMetrics', {}),
                'heatmapRegions': result.get('heatmapRegions', []),
            }
            await db.medical_history.insert_one(entry)
        except Exception as e:
            # Log and continue returning analysis result
            print('DB insert failed (non-fatal):', e)

    return JSONResponse(result)


@app.get('/api/medical-history/{username}')
async def get_history(username: str):
    cursor = db.medical_history.find({'username': username}).sort('date', -1).limit(50)
    items = []
    async for doc in cursor:
        doc['_id'] = str(doc['_id'])
        items.append(doc)
    return {'success': True, 'data': items}


@app.get('/api/db-health')
async def db_health():
    """Simple health check endpoint for MongoDB connectivity."""
    try:
        await client.admin.command('ping')
        return {'connected': True, 'db': 'lungdx'}
    except Exception as e:
        return JSONResponse({'connected': False, 'error': str(e)}, status_code=503)


@app.put("/api/profile/{username}")
async def update_profile(username: str, req: UpdateProfileRequest):
    user = await db.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    update_data = {k: v for k, v in req.dict().items() if v is not None}
    
    if update_data:
        await db.users.update_one({"username": username}, {"$set": update_data})
        
    updated_user = await db.users.find_one({"username": username})
    
    return {
        "success": True, 
        "user": {
            "username": updated_user['username'],
            "firstName": updated_user.get('firstName', ''),
            "lastName": updated_user.get('lastName', ''),
            "email": updated_user.get('email', ''),
            "role": updated_user.get('role', 'user'),
            "phoneNumber": updated_user.get('phoneNumber'),
            "dateOfBirth": updated_user.get('dateOfBirth'),
            "gender": updated_user.get('gender'),
            "profilePicture": updated_user.get('profilePicture')
        }
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), reload=True)
