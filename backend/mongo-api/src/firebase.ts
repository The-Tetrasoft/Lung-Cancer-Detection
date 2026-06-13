import admin from 'firebase-admin';
import { getFirestore } from 'firebase-admin/firestore';

// Initialize Firebase Admin
let db: FirebaseFirestore.Firestore | null = null;

/**
 * Initialize Firebase Admin SDK and Firestore.
 * - Supports service account JSON via FIREBASE_SERVICE_ACCOUNT_PATH or a default file in this repo.
 * - Supports credentials via FIREBASE_PROJECT_ID / FIREBASE_CLIENT_EMAIL / FIREBASE_PRIVATE_KEY env vars.
 * - Supports emulator mode when FIREBASE_USE_EMULATOR=true and FIRESTORE_EMULATOR_HOST is provided (or defaults to localhost:8080).
 */
export async function initializeFirebase() {
  try {
    if (admin.apps && admin.apps.length > 0) {
      // already initialized in this process
      db = getFirestore();
      console.log('ℹ️ Firebase already initialized in this process');
      return db;
    }

    const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH;
    const defaultLocalPath = './lungdx-9b2c3-firebase-adminsdk-fbsvc-0c134ba37d.json';

    console.log('Current working directory:', process.cwd());
    console.log('Looking for service account at:', defaultLocalPath);

    // Helper to check for the default file
    const hasDefaultFile = await (async () => {
      try {
        const { access } = await import('fs/promises');
        await access(defaultLocalPath);
        return true;
      } catch {
        return false;
      }
    })();

    console.log('Has default file:', hasDefaultFile);

    // Emulator support
    if (process.env.FIREBASE_USE_EMULATOR === 'true') {
      const emulatorHost = process.env.FIRESTORE_EMULATOR_HOST || 'localhost:8080';
      // Ensure variable is set for underlying Firestore clients
      process.env.FIRESTORE_EMULATOR_HOST = emulatorHost;
      const projectId = process.env.FIREBASE_PROJECT_ID || 'demo-project';
      admin.initializeApp({ projectId });
      db = getFirestore();
      console.log(`✅ Firebase Admin initialized for emulator at ${emulatorHost} (project: ${projectId})`);
    } else if (serviceAccountPath || hasDefaultFile) {
      // initialize using service account JSON file
      try {
        const { readFileSync } = await import('fs');
        const { resolve } = await import('path');
        const fullPath = resolve(serviceAccountPath || defaultLocalPath);
        console.log('Resolved full path:', fullPath);
        const serviceAccountJson = readFileSync(fullPath, 'utf8');
        const serviceAccount = JSON.parse(serviceAccountJson);

        admin.initializeApp({
          credential: admin.credential.cert(serviceAccount),
        });
        db = getFirestore();
        console.log(`✅ Firebase Admin initialized using service account file: ${fullPath}`);
      } catch (fileError) {
        console.error('❌ Error loading service account file:', fileError);
        db = null;
      }
    } else {
      // Try env var credentials (useful for cloud deployments)
      const projectId = process.env.FIREBASE_PROJECT_ID;
      const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
      const privateKeyRaw = process.env.FIREBASE_PRIVATE_KEY;

      if (projectId && clientEmail && privateKeyRaw) {
        // restore newlines in private key
        const privateKey = privateKeyRaw.replace(/\\n/g, '\n');
        admin.initializeApp({
          credential: admin.credential.cert({ projectId, clientEmail, privateKey }),
        });
        db = getFirestore();
        console.log('✅ Firebase Admin initialized using environment credentials');
      } else {
        console.warn('⚠️ Firebase credentials not found (service account file or env vars).');
        console.warn('   To enable Firestore, set FIREBASE_SERVICE_ACCOUNT_PATH or FIREBASE_PROJECT_ID/FIREBASE_CLIENT_EMAIL/FIREBASE_PRIVATE_KEY, or set FIREBASE_USE_EMULATOR=true to use emulator.');
        db = null;
      }
    }

    // Quick runtime test to ensure Firestore is reachable.
    if (db) {
      try {
        const collections = await db.listCollections();
        console.log(`ℹ️ Firestore reachable — found ${collections.length} top-level collections`);
      } catch (testErr) {
        console.error('❌ Firestore test failed (could not list collections):', testErr);
        // keep db as null to indicate failure
        db = null;
      }
    }

    return db;
  } catch (error) {
    console.error('❌ Firebase initialization error:', error);
    db = null;
    return null;
  }
}

export function getDb(): FirebaseFirestore.Firestore | null {
  return db;
}

export function isFirebaseConnected(): boolean {
  return db !== null;
}

// Helper functions for common operations
export const FirebaseHelper = {
  // Get a document by ID
  async getDoc(collection: string, id: string) {
    if (!db) return null;
    const docRef = db.collection(collection).doc(id);
    const doc = await docRef.get();
    return doc.exists ? { id: doc.id, ...doc.data() } : null;
  },

  // Get documents by query
  async getDocs(collection: string, filters?: { field: string; operator: FirebaseFirestore.WhereFilterOp; value: any }[]) {
    if (!db) return [];
    let query: FirebaseFirestore.Query = db.collection(collection);

    if (filters) {
      filters.forEach(filter => {
        query = query.where(filter.field, filter.operator, filter.value);
      });
    }

    const snapshot = await query.get();
    return snapshot.docs.map(doc => ({ id: doc.id, ...doc.data() }));
  },

  // Create a document
  async createDoc(collection: string, data: any, customId?: string) {
    if (!db) return null;
    if (customId) {
      await db.collection(collection).doc(customId).set(data);
      return { id: customId, ...data };
    } else {
      const docRef = await db.collection(collection).add(data);
      return { id: docRef.id, ...data };
    }
  },

  // Update a document
  async updateDoc(collection: string, id: string, data: any) {
    if (!db) return null;
    await db.collection(collection).doc(id).update(data);
    return { id, ...data };
  },

  // Delete a document
  async deleteDoc(collection: string, id: string) {
    if (!db) return false;
    await db.collection(collection).doc(id).delete();
    return true;
  },

  // Get document by field value
  async findOne(collection: string, field: string, value: any) {
    if (!db) return null;
    const snapshot = await db.collection(collection).where(field, '==', value).limit(1).get();
    if (snapshot.empty) return null;
    const doc = snapshot.docs[0];
    return { id: doc.id, ...doc.data() };
  },

  // Check if document exists
  async exists(collection: string, field: string, value: any) {
    const doc = await this.findOne(collection, field, value);
    return doc !== null;
  },
};
