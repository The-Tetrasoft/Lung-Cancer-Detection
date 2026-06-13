# LungDx Backend API (Firebase Firestore)

This backend provides REST endpoints for the LungDx frontend using Firebase Firestore as the database.

## 🔥 Features

- **Firebase Firestore** - Serverless NoSQL database
- **Express + TypeScript** - Modern Node.js backend
- **In-Memory Fallback** - Works without Firebase for testing
- **File Upload** - Multer for X-ray image handling
- **CORS Enabled** - Cross-origin requests supported

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Firebase

#### Option A: Use Service Account JSON (Recommended for Local)

1. Download your Firebase service account JSON from Firebase Console
2. Save it in this directory (e.g., `serviceAccountKey.json`)
3. Create `.env` file:

```env
FIREBASE_SERVICE_ACCOUNT_PATH=./serviceAccountKey.json
PORT=8000
PY_BACKEND_BASE=http://localhost:8100
```

#### Option B: Use Environment Variables (For Deployment)

Create `.env` file:

```env
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nYour-Key-Here\n-----END PRIVATE KEY-----\n"
PORT=8000
PY_BACKEND_BASE=http://localhost:8100
```

### 3. Start Development Server

```bash
npm run dev
```

You should see:
```
✅ Firebase initialized successfully
🚀 Mongo API (Firebase) listening on :8000
📊 Database: ✅ Firebase Firestore
```

---

## 📦 API Endpoints

### Authentication
- `POST /api/signup` - Create new user account
- `POST /api/login` - User login
- `POST /api/check-username` - Check username availability

### Profile
- `PUT /api/profile/:username` - Update user profile

### Medical History
- `POST /api/history` - Create medical record
- `GET /api/history/:username` - Get user's medical history

### Analysis
- `POST /api/analyze` - Analyze X-ray image (multipart/form-data)

### Health Check
- `GET /api/health` - Server and database status

---

## 🗄️ Firestore Collections

### `users`
```javascript
{
  username: string,
  firstName: string,
  lastName: string,
  email: string,
  password: string (base64 encoded),
  dateOfBirth: string,
  gender: string,
  phoneNumber: string,
  profilePicture: string,
  role: string,
  created_at: string (ISO date),
  loginAttempts: number
}
```

### `medicalRecords`
```javascript
{
  username: string,
  date: string (ISO date),
  type: string,
  result: string,
  confidence: number,
  notes: string,
  doctorName: string,
  imageUrl: string,
  raw: object
}
```

---

## 🧪 Demo Mode (No Firebase)

If Firebase credentials are not configured, the server runs in **demo mode** with in-memory storage:

```
⚠️  Running without Firebase. Using in-memory storage for demo.
🚀 Mongo API (Firebase) listening on :8000
📊 Database: ⚠️  In-Memory (Demo Mode)
```

**Note:** Data is not persistent in demo mode and will be lost on server restart.

---

## 🔧 Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build TypeScript to JavaScript
- `npm start` - Start production server (requires build first)

---

## 📚 Firebase Setup Guide

For detailed Firebase setup instructions, see:
- `../../FIREBASE_SETUP.md` - Complete setup guide
- `../../QUICKSTART_FIREBASE.md` - Quick reference

---

## 🔐 Security Notes

### Current Implementation
- ✅ Base64 password encoding (upgrade to bcrypt recommended)
- ✅ Login attempt limiting (5 max)
- ✅ CORS configuration
- ✅ Input validation

### For Production
- ⚠️ Implement bcrypt password hashing
- ⚠️ Add JWT authentication
- ⚠️ Configure Firestore security rules
- ⚠️ Enable HTTPS only
- ⚠️ Add rate limiting
- ⚠️ Implement audit logging

---

## 🐛 Troubleshooting

### Server shows "In-Memory (Demo Mode)"
**Solution:** Firebase credentials not configured
- Check `.env` file exists
- Verify `FIREBASE_SERVICE_ACCOUNT_PATH` points to correct JSON file
- Ensure JSON file has proper permissions
- Restart server after updating `.env`

### Error: "Firebase initialization error"
**Solution:** Check credentials
- Verify service account JSON is valid
- Check environment variables are set correctly
- Ensure Firebase project exists

### Port already in use
**Solution:** Change port in `.env` or kill existing process
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill
```

---

## 📊 Dependencies

### Core
- `express` - Web framework
- `firebase-admin` - Firebase Admin SDK
- `cors` - CORS middleware
- `dotenv` - Environment variables
- `multer` - File upload handling

### Development
- `typescript` - Type safety
- `tsx` - TypeScript execution
- `@types/*` - Type definitions

---

## 🎉 Success!

Your LungDx backend is now running with Firebase Firestore!

**Next Steps:**
1. Test API endpoints
2. Configure Firestore security rules
3. Deploy to production (Vercel, Railway, Render, etc.)

---

**Happy Coding! 🚀**
