import { initializeApp } from 'firebase/app';
import { getAuth, setPersistence, browserLocalPersistence, indexedDBLocalPersistence } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import { firebaseEnv } from './firebase-env';

// Use the direct configuration from firebase-env.ts
const firebaseConfig = firebaseEnv;

// Check for missing configuration
if (firebaseConfig.apiKey === "REPLACE_WITH_YOUR_API_KEY" || !firebaseConfig.apiKey) {
    console.error("❌ Firebase API Key is missing! Please check src/firebase-env.ts");
}

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Auth with Persistence
const auth = getAuth(app);

// Use indexedDB for persistence (works on Android/iOS/Web)
// This ensures the user stays logged in even after app restart
setPersistence(auth, indexedDBLocalPersistence)
    .catch((error) => {
        console.error("Auth Persistence Error:", error);
    });

// Initialize Firestore
const db = getFirestore(app);

// Initialize Storage
const storage = getStorage(app);

export { auth, db, storage };
