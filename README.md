# LungDx: AI-Powered Lung Cancer Detection Application

LungDx is a cross-platform (Web & Mobile Android) application designed to facilitate early detection and monitoring of lung cancer. By analyzing lung X-ray scans with a Convolutional Neural Network (CNN) model, patients can perform self-risk assessments, log their diagnostic history, and book consultations with specialized doctors. The application also provides an administrative portal for hospital staff to oversee doctors, user accounts, appointment schedules, and analytics.

---

## Key Features

### 1. Patient Portal
* **AI Diagnostic Inference:** Upload chest X-rays to get a real-time risk assessment (Low, Medium, High, or Critical risk level) along with confidence metrics, lung opacity assessments, and medical recommendations.
* **Specialist Doctor Registry:** Search and filter specialist doctors by name, hospital affiliation, location, or clinical specialties.
* **Appointment Scheduling:** Book specific time slots for consultations with recommended specialists.
* **Personal Medical History:** Access previous X-ray analysis logs, including risk trends and diagnostic stats.

### 2. Admin Dashboard
* **Doctor Profile Management:** Add new specialists, configure consultation fees, experience, contact details, and hospital locations.
* **User Accounts Panel:** Review registered users, manage roles, and delete inactive profiles.
* **Appointment Coordination:** Track, approve, reschedule, or cancel scheduled appointments.
* **System Analytics:** Visually monitor health metrics, case distributions, and risk patterns using interactive charts.

### 3. System Resilience & Failover Mechanics
* **Database Failover:** If Firebase Firestore is unreachable, the Node.js Express server seamlessly degrades to an **In-Memory Store Map** to keep the application functional for demonstration and testing.
* **ML Service Offline Fallback:** If the Python ML server is down, the Express gateway generates **synthetic analysis reports** with realistic bounding boxes and classifications to ensure a continuous client user experience.

---

## Technical Architecture

The codebase is structured as a decoupled web application with a client frontend, an Express API gateway, and a Python machine learning engine:

```
├── android/                  # Native Android wrapper assets (Capacitor)
├── backend/
│   ├── model/                # CNN model loader and pre-processing
│   ├── mongo-api/            # Node.js + Express API Gateway (TypeScript)
│   └── app.py                # FastAPI Python ML Server
├── src/                      # React + TypeScript Client Frontend
│   ├── components/           # Modularized UI Views & Layouts
│   ├── utils/                # Config, persistence, and mobile bridges
│   └── App.tsx               # Client state routing & Auth listeners
└── capacitor.config.ts       # Capacitor native integration config
```

### Technology Stack
* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, Radix UI Primitives, Recharts (Analytics), Sonner (Toasts)
* **Mobile Shell:** Capacitor 7
* **API Gateway:** Express.js, Node.js, TypeScript, Multer, Firebase Admin SDK
* **ML Server:** FastAPI, Python 3, DenseNet169 (TFLite Interpreter & PyTorch), Pillow (PIL), NumPy, Motor (Async MongoDB Driver)
* **Databases:** Firebase Firestore (Cloud Database), MongoDB (ML Logging Data)

---

## Setup & Running Locally

### Prerequisites
* **Node.js** (v18 or higher)
* **Python** (3.8 - 3.11 recommended)
* **Java SDK & Android Studio** (Only if building the native mobile app)

---

### 1. Run the Frontend Client
Navigate to the root directory and install dependencies:
```bash
npm install
```

Configure your Firebase environment credentials in `src/firebase-env.ts`.

Run the client development server:
```bash
npm run dev
```
The client will be running on `http://localhost:5173`.

---

### 2. Run the Node.js API Gateway
Navigate to the API server directory:
```bash
cd backend/mongo-api
npm install
```

Create a `.env` file in `backend/mongo-api/.env` with the following configuration:
```env
PORT=8000
PY_BACKEND_BASE=http://localhost:8100
FIREBASE_SERVICE_ACCOUNT_PATH=./your-firebase-adminsdk-cert.json
```

Compile TypeScript and start the server:
```bash
# Build the TypeScript codebase
npm run build

# Start the node server
npm start
```
The Express gateway will start on `http://localhost:8000`.

---

### 3. Run the Python ML Engine
Navigate to the backend directory:
```bash
cd backend
```

Create a Python virtual environment and activate it:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
python app.py
```
The ML service will run on `http://localhost:8100` (or the default port mapped via the environment).

---

### 4. Build and Run the Android App (Capacitor)
Ensure your Android SDK and paths are correctly configured. From the root directory:

1. Build the production web bundle:
   ```bash
   npm run build
   ```
2. Sync the compiled assets to the native Android folder:
   ```bash
   npm run cap:sync
   ```
3. Compile and build the debug APK using Gradle:
   ```bash
   npm run cap:build:android
   ```
4. Open the Android project in Android Studio to run on a physical device or emulator:
   ```bash
   npm run cap:android
   ```
