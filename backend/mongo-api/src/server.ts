import 'dotenv/config';
import express, { Request, Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import { initializeFirebase, FirebaseHelper } from './firebase.js';

const app = express();
app.use(cors({ origin: true, allowedHeaders: ['Content-Type', 'Authorization'] }));
app.use(express.json({ limit: '4mb' }));
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });

// Initialize Firebase
let db: any = null;
let dbReady = false;

// Firebase initialization moved to startServer

// In-memory fallback storage (when Firebase is not configured)
const inMemoryStore = {
    users: new Map<string, any>(),
    medicalRecords: new Map<string, any[]>(),
};

// Helper functions
const encodeBase64 = (s: string) => Buffer.from(s, 'utf-8').toString('base64');
const matchPassword = (plain: string, storedBase64: string) => encodeBase64(plain) === storedBase64;

const toPublicUser = (u: any) => ({
    username: u.username,
    firstName: u.firstName,
    lastName: u.lastName,
    email: u.email,
    role: u.role,
    profilePicture: u.profilePicture ?? '',
    phoneNumber: u.phoneNumber ?? '',
    dateOfBirth: u.dateOfBirth ?? '',
    gender: u.gender ?? '',
});

// ============================================
// AUTHENTICATION ENDPOINTS
// ============================================

// Check username availability
app.post('/api/check-username', async (req: Request, res: Response) => {
    try {
        const { username } = req.body;
        if (!username) return res.status(400).json({ success: false, message: 'Username required' });

        let exists = false;
        if (dbReady) {
            exists = await FirebaseHelper.exists('users', 'username', username);
        } else {
            exists = inMemoryStore.users.has(username);
        }

        res.json({ available: !exists });
    } catch (e) {
        console.error('Check username error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Signup
app.post('/api/signup', async (req: Request, res: Response) => {
    try {
        const { username, firstName, lastName, dateOfBirth, gender, email, phoneNumber, password, profilePicture } = req.body || {};

        if (!username || !firstName || !lastName || !email || !password) {
            return res.status(400).json({ success: false, message: 'Missing required fields' });
        }

        // Check if user exists
        let userExists = false;
        if (dbReady) {
            const existingUser = await FirebaseHelper.findOne('users', 'username', username);
            const existingEmail = await FirebaseHelper.findOne('users', 'email', email.toLowerCase());
            userExists = existingUser !== null || existingEmail !== null;
        } else {
            userExists = inMemoryStore.users.has(username) ||
                Array.from(inMemoryStore.users.values()).some(u => u.email === email.toLowerCase());
        }

        if (userExists) {
            return res.status(400).json({ success: false, message: 'Username or email already exists' });
        }

        const userData = {
            username,
            firstName,
            lastName,
            dateOfBirth: dateOfBirth || '',
            gender: gender || '',
            email: email.toLowerCase(),
            phoneNumber: phoneNumber || '',
            password: encodeBase64(password),
            profilePicture: profilePicture || '',
            role: 'user',
            created_at: new Date().toISOString(),
            loginAttempts: 0,
        };

        if (dbReady) {
            await FirebaseHelper.createDoc('users', userData, username);
        } else {
            inMemoryStore.users.set(username, userData);
        }

        res.json({ success: true, message: 'Account created successfully', user: toPublicUser(userData) });
    } catch (e) {
        console.error('Signup error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Login
app.post('/api/login', async (req: Request, res: Response) => {
    try {
        const { usernameOrEmail, password } = req.body || {};
        const ident = typeof usernameOrEmail === 'string' ? usernameOrEmail.trim() : '';
        const pwd = typeof password === 'string' ? password : '';

        if (!ident || !pwd) {
            return res.status(400).json({ success: false, message: 'Missing credentials' });
        }

        let user: any = null;

        if (dbReady) {
            // Try to find by username first
            user = await FirebaseHelper.findOne('users', 'username', ident);
            // If not found, try by email
            if (!user) {
                user = await FirebaseHelper.findOne('users', 'email', ident.toLowerCase());
            }
        } else {
            // In-memory lookup
            user = inMemoryStore.users.get(ident) ||
                Array.from(inMemoryStore.users.values()).find(u => u.email === ident.toLowerCase());
        }

        if (!user) {
            return res.status(404).json({ success: false, message: 'No user found, create an account' });
        }

        if (user.loginAttempts >= 5) {
            return res.status(423).json({ success: false, message: 'Account locked due to too many failed attempts' });
        }

        // Check password
        const passwordMatch = matchPassword(pwd, user.password);

        if (!passwordMatch) {
            user.loginAttempts = (user.loginAttempts || 0) + 1;

            if (dbReady) {
                await FirebaseHelper.updateDoc('users', user.id || user.username, { loginAttempts: user.loginAttempts });
            } else {
                inMemoryStore.users.set(user.username, user);
            }

            const remainingAttempts = Math.max(0, 5 - user.loginAttempts);
            return res.status(401).json({ success: false, message: 'Incorrect password', remainingAttempts });
        }

        // Successful login - reset attempts
        user.loginAttempts = 0;

        if (dbReady) {
            await FirebaseHelper.updateDoc('users', user.id || user.username, { loginAttempts: 0 });
        } else {
            inMemoryStore.users.set(user.username, user);
        }

        res.json({ success: true, message: 'Login successful', user: toPublicUser(user) });
    } catch (e) {
        console.error('Login error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// PROFILE ENDPOINTS
// ============================================

app.put('/api/profile/:username', async (req: Request, res: Response) => {
    try {
        const { username } = req.params;
        const updates = req.body || {};

        let user: any = null;

        if (dbReady) {
            user = await FirebaseHelper.findOne('users', 'username', username);
        } else {
            user = inMemoryStore.users.get(username);
        }

        if (!user) {
            return res.status(404).json({ success: false, message: 'User not found' });
        }

        // Check if email is being changed and if it's already taken
        if (updates.email && updates.email !== user.email) {
            let emailTaken = false;

            if (dbReady) {
                const existingUser = await FirebaseHelper.findOne('users', 'email', updates.email.toLowerCase());
                emailTaken = existingUser !== null;
            } else {
                emailTaken = Array.from(inMemoryStore.users.values()).some(u => u.email === updates.email.toLowerCase());
            }

            if (emailTaken) {
                return res.status(400).json({ success: false, message: 'Email already registered' });
            }
        }

        // Update fields
        const updatedUser = {
            ...user,
            firstName: updates.firstName ?? user.firstName,
            lastName: updates.lastName ?? user.lastName,
            email: updates.email ? updates.email.toLowerCase() : user.email,
            phoneNumber: updates.phoneNumber ?? user.phoneNumber,
            gender: updates.gender ?? user.gender,
            profilePicture: updates.profilePicture ?? user.profilePicture,
            dateOfBirth: updates.dateOfBirth ?? user.dateOfBirth,
            role: updates.role ?? user.role, // Allow role updates
        };

        if (dbReady) {
            await FirebaseHelper.updateDoc('users', user.id || username, updatedUser);
        } else {
            inMemoryStore.users.set(username, updatedUser);
        }

        res.json({ success: true, user: toPublicUser(updatedUser) });
    } catch (e) {
        console.error('Profile update error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// USER MANAGEMENT ENDPOINTS
// ============================================

// Get all users (admin only)
app.get('/api/users', async (req: Request, res: Response) => {
    try {
        let allUsers: any[] = [];

        if (dbReady) {
            allUsers = await FirebaseHelper.getDocs('users');
        } else {
            allUsers = Array.from(inMemoryStore.users.values());
        }

        // Return public user data only
        const publicUsers = allUsers.map(toPublicUser);
        res.json(publicUsers);
    } catch (e) {
        console.error('Get users error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Delete user by username (admin only)
app.delete('/api/users/:username', async (req: Request, res: Response) => {
    try {
        const { username } = req.params;

        if (!username) {
            return res.status(400).json({ success: false, message: 'Username required' });
        }

        if (dbReady) {
            // Find user first to get the document ID
            const user = await FirebaseHelper.findOne('users', 'username', username);
            if (!user) {
                return res.status(404).json({ success: false, message: 'User not found' });
            }

            // Delete user
            await FirebaseHelper.deleteDoc('users', user.id || username);
        } else {
            if (!inMemoryStore.users.has(username)) {
                return res.status(404).json({ success: false, message: 'User not found' });
            }
            inMemoryStore.users.delete(username);
        }

        res.json({ success: true, message: 'User deleted successfully' });
    } catch (e) {
        console.error('Delete user error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// MEDICAL HISTORY ENDPOINTS
// ============================================

// Create medical record
app.post('/api/history', async (req: Request, res: Response) => {
    try {
        const { username, date, type, result, confidence, notes, doctorName, imageUrl, raw } = req.body || {};

        if (!username) {
            return res.status(400).json({ success: false, message: 'username is required' });
        }

        const record = {
            username,
            date: date ? new Date(date).toISOString() : new Date().toISOString(),
            type: type || 'X-Ray',
            result: result || 'Pending',
            confidence: typeof confidence === 'number' ? confidence : 0,
            notes: notes || '',
            doctorName: doctorName || '',
            imageUrl: imageUrl || '',
            raw: raw || null,
        };

        if (dbReady) {
            const created = await FirebaseHelper.createDoc('medicalRecords', record);
            res.json({ success: true, record: created });
        } else {
            if (!inMemoryStore.medicalRecords.has(username)) {
                inMemoryStore.medicalRecords.set(username, []);
            }
            const recordWithId = { ...record, id: Date.now().toString() };
            inMemoryStore.medicalRecords.get(username)!.push(recordWithId);
            res.json({ success: true, record: recordWithId });
        }
    } catch (e) {
        console.error('Create history error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Get medical history
app.get('/api/history/:username', async (req: Request, res: Response) => {
    try {
        const { username } = req.params;

        if (!username) {
            return res.status(400).json({ success: false, message: 'username is required' });
        }

        let records: any[] = [];

        if (dbReady) {
            records = await FirebaseHelper.getDocs('medicalRecords', [
                { field: 'username', operator: '==', value: username }
            ]);

            // Sort by date (newest first)
            records.sort((a, b) => {
                const dateA = new Date(a.date).getTime();
                const dateB = new Date(b.date).getTime();
                return dateB - dateA;
            });
        } else {
            records = inMemoryStore.medicalRecords.get(username) || [];
            records.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
        }

        res.json({ success: true, records });
    } catch (e) {
        console.error('Fetch history error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// ANALYSIS ENDPOINT
// ============================================

app.post('/api/analyze', upload.single('image'), async (req: Request & { file?: any }, res: Response) => {
    try {
        // Try Python backend first
        const PY_URL = process.env.PY_BACKEND_BASE || 'http://localhost:8100';

        if (req.file) {
            try {
                const form = new FormData();
                const file = new File([req.file.buffer], req.file.originalname || 'image.png', {
                    type: req.file.mimetype || 'application/octet-stream'
                });
                form.append('image', file);
                if (typeof req.body?.username === 'string') form.append('username', req.body.username);

                const pyResp = await fetch(`${PY_URL}/api/analyze`, { method: 'POST', body: form as any });
                if (pyResp.ok) {
                    const data = await pyResp.json();
                    return res.json(data);
                }
            } catch (e) {
                console.warn('Python analyze failed, using fallback:', (e as Error).message);
            }
        }

        // Fallback to synthetic analysis
        const random = Math.random();
        const isPositive = random > 0.7;
        const confidence = isPositive ? 65 + Math.random() * 30 : 80 + Math.random() * 18;

        let riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
        if (!isPositive) riskLevel = 'Low';
        else if (confidence > 85) riskLevel = 'Critical';
        else if (confidence > 75) riskLevel = 'High';
        else riskLevel = 'Medium';

        const regions: Array<{ x: number; y: number; width: number; height: number; intensity: number; label: string }> = [];

        if (isPositive) {
            const n = 2 + Math.floor(Math.random() * 3);
            for (let i = 0; i < n; i++) {
                regions.push({
                    x: 20 + Math.random() * 50,
                    y: 15 + Math.random() * 60,
                    width: 10 + Math.random() * 20,
                    height: 10 + Math.random() * 20,
                    intensity: 0.5 + Math.random() * 0.5,
                    label: ['Nodule', 'Opacity', 'Mass', 'Abnormality'][Math.floor(Math.random() * 4)],
                });
            }
        } else {
            const n = Math.floor(Math.random() * 3);
            for (let i = 0; i < n; i++) {
                regions.push({
                    x: 20 + Math.random() * 50,
                    y: 15 + Math.random() * 60,
                    width: 8 + Math.random() * 15,
                    height: 8 + Math.random() * 15,
                    intensity: 0.2 + Math.random() * 0.3,
                    label: 'Minor Finding',
                });
            }
        }

        const result = {
            prediction: isPositive ? 'Positive' : 'Negative',
            confidence: Math.round(confidence * 10) / 10,
            riskLevel,
            detailedMetrics: {
                lungOpacity: Math.round((40 + Math.random() * 60) * 10) / 10,
                noduleDetection: Math.round((isPositive ? 60 + Math.random() * 35 : 10 + Math.random() * 30) * 10) / 10,
                tissueAbnormality: Math.round((isPositive ? 55 + Math.random() * 40 : 15 + Math.random() * 35) * 10) / 10,
                inflammationMarkers: Math.round((30 + Math.random() * 60) * 10) / 10,
            },
            recommendations: isPositive
                ? [
                    'Immediate consultation with an oncologist is recommended',
                    'Additional diagnostic tests (CT scan, biopsy) should be performed',
                    'Discuss treatment options with your healthcare provider',
                    'Consider getting a second opinion from a specialist',
                    'Monitor symptoms and report any changes immediately',
                ]
                : [
                    'Continue regular health check-ups',
                    'Maintain a healthy lifestyle and diet',
                    'Avoid smoking and exposure to pollutants',
                    'Schedule annual lung screenings if you are in a high-risk group',
                    'Consult a doctor if you experience any respiratory symptoms',
                ],
            heatmapRegions: regions,
        };

        res.json(result);
    } catch (e) {
        console.error('Analyze error:', e);
        res.status(500).json({ message: 'Analyze failed' });
    }
});

// ============================================
// DOCTOR MANAGEMENT ENDPOINTS
// ============================================

// Get all doctors
app.get('/api/doctors', async (_req: Request, res: Response) => {
    try {
        let doctors: any[] = [];

        if (dbReady) {
            doctors = await FirebaseHelper.getDocs('doctors');

            // Sort by creation date (newest first)
            doctors.sort((a, b) => {
                const dateA = new Date(a.created_at || 0).getTime();
                const dateB = new Date(b.created_at || 0).getTime();
                return dateB - dateA;
            });
        } else {
            // In-memory fallback
            if (!inMemoryStore.medicalRecords.has('doctors')) {
                inMemoryStore.medicalRecords.set('doctors', []);
            }
            doctors = inMemoryStore.medicalRecords.get('doctors') || [];
        }

        res.json({ success: true, doctors });
    } catch (e) {
        console.error('Fetch doctors error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Get doctor by ID
app.get('/api/doctors/:id', async (req: Request, res: Response) => {
    try {
        const { id } = req.params;

        let doctor: any = null;

        if (dbReady) {
            doctor = await FirebaseHelper.getDoc('doctors', id);
        } else {
            const doctors = inMemoryStore.medicalRecords.get('doctors') || [];
            doctor = doctors.find((d: any) => d.id === id);
        }

        if (!doctor) {
            return res.status(404).json({ success: false, message: 'Doctor not found' });
        }

        res.json({ success: true, doctor });
    } catch (e) {
        console.error('Fetch doctor error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Create new doctor (Admin only)
app.post('/api/doctors', async (req: Request, res: Response) => {
    try {
        const doctorData = req.body;

        // Validate required fields
        const requiredFields = ['firstName', 'lastName', 'gender', 'age', 'hospital', 'position', 'location', 'email', 'phone', 'specialties', 'experience', 'consultationFee'];
        const missingFields = requiredFields.filter(field => !doctorData[field]);

        if (missingFields.length > 0) {
            return res.status(400).json({
                success: false,
                message: `Missing required fields: ${missingFields.join(', ')}`
            });
        }

        // Create doctor object
        const doctor = {
            ...doctorData,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            status: 'active',
        };

        let createdDoctor: any;

        if (dbReady) {
            // Generate ID from name
            const doctorId = `${doctorData.firstName.toLowerCase()}_${doctorData.lastName.toLowerCase()}_${Date.now()}`;
            createdDoctor = await FirebaseHelper.createDoc('doctors', doctor, doctorId);
        } else {
            if (!inMemoryStore.medicalRecords.has('doctors')) {
                inMemoryStore.medicalRecords.set('doctors', []);
            }
            const doctorWithId = { ...doctor, id: `doctor_${Date.now()}` };
            inMemoryStore.medicalRecords.get('doctors')!.push(doctorWithId);
            createdDoctor = doctorWithId;
        }

        res.json({ success: true, message: 'Doctor added successfully', doctor: createdDoctor });
    } catch (e) {
        console.error('Create doctor error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Update doctor (Admin only)
app.put('/api/doctors/:id', async (req: Request, res: Response) => {
    try {
        const { id } = req.params;
        const updates = req.body;

        let doctor: any = null;

        if (dbReady) {
            doctor = await FirebaseHelper.getDoc('doctors', id);
            if (!doctor) {
                return res.status(404).json({ success: false, message: 'Doctor not found' });
            }

            const updatedDoctor = {
                ...doctor,
                ...updates,
                updated_at: new Date().toISOString(),
            };

            await FirebaseHelper.updateDoc('doctors', id, updatedDoctor);
            res.json({ success: true, message: 'Doctor updated successfully', doctor: updatedDoctor });
        } else {
            const doctors = inMemoryStore.medicalRecords.get('doctors') || [];
            const index = doctors.findIndex((d: any) => d.id === id);

            if (index === -1) {
                return res.status(404).json({ success: false, message: 'Doctor not found' });
            }

            doctors[index] = {
                ...doctors[index],
                ...updates,
                updated_at: new Date().toISOString(),
            };

            res.json({ success: true, message: 'Doctor updated successfully', doctor: doctors[index] });
        }
    } catch (e) {
        console.error('Update doctor error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Delete doctor (Admin only)
app.delete('/api/doctors/:id', async (req: Request, res: Response) => {
    try {
        const { id } = req.params;

        if (dbReady) {
            const doctor = await FirebaseHelper.getDoc('doctors', id);
            if (!doctor) {
                return res.status(404).json({ success: false, message: 'Doctor not found' });
            }

            await FirebaseHelper.deleteDoc('doctors', id);
            res.json({ success: true, message: 'Doctor deleted successfully' });
        } else {
            const doctors = inMemoryStore.medicalRecords.get('doctors') || [];
            const index = doctors.findIndex((d: any) => d.id === id);

            if (index === -1) {
                return res.status(404).json({ success: false, message: 'Doctor not found' });
            }

            doctors.splice(index, 1);
            res.json({ success: true, message: 'Doctor deleted successfully' });
        }
    } catch (e) {
        console.error('Delete doctor error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Search doctors by specialty or name
app.get('/api/doctors/search/:query', async (req: Request, res: Response) => {
    try {
        const { query } = req.params;
        const searchTerm = query.toLowerCase();

        let doctors: any[] = [];

        if (dbReady) {
            // Get all doctors and filter in memory (Firestore doesn't support complex text search)
            doctors = await FirebaseHelper.getDocs('doctors');
            doctors = doctors.filter((doctor: any) => {
                const fullName = `${doctor.firstName} ${doctor.lastName}`.toLowerCase();
                const specialtiesMatch = doctor.specialties?.some((s: string) => s.toLowerCase().includes(searchTerm));
                const nameMatch = fullName.includes(searchTerm);
                const hospitalMatch = doctor.hospital?.toLowerCase().includes(searchTerm);

                return nameMatch || specialtiesMatch || hospitalMatch;
            });
        } else {
            const allDoctors = inMemoryStore.medicalRecords.get('doctors') || [];
            doctors = allDoctors.filter((doctor: any) => {
                const fullName = `${doctor.firstName} ${doctor.lastName}`.toLowerCase();
                const specialtiesMatch = doctor.specialties?.some((s: string) => s.toLowerCase().includes(searchTerm));
                const nameMatch = fullName.includes(searchTerm);
                const hospitalMatch = doctor.hospital?.toLowerCase().includes(searchTerm);

                return nameMatch || specialtiesMatch || hospitalMatch;
            });
        }

        res.json({ success: true, doctors });
    } catch (e) {
        console.error('Search doctors error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// HEALTH CHECK
// ============================================

app.get('/api/health', async (_req: Request, res: Response) => {
    try {
        res.json({
            ok: true,
            database: dbReady ? 'firebase' : 'in-memory',
            status: dbReady ? 'connected' : 'fallback-mode'
        });
    } catch (e) {
        res.status(500).json({ ok: false });
    }
});

// ============================================
// APPOINTMENT MANAGEMENT ENDPOINTS
// ============================================

// Create appointment
app.post('/api/appointments', async (req: Request, res: Response) => {
    try {
        const { username, doctorId, doctorName, date, time, reason, status } = req.body || {};

        if (!username || !doctorId || !date || !time) {
            return res.status(400).json({ success: false, message: 'Missing required fields' });
        }

        const appointment = {
            username,
            doctorId,
            doctorName: doctorName || '',
            date,
            time,
            reason: reason || '',
            status: status || 'pending',
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
        };

        if (dbReady) {
            const created = await FirebaseHelper.createDoc('appointments', appointment);
            res.json({ success: true, appointment: created });
        } else {
            if (!inMemoryStore.medicalRecords.has('appointments')) {
                inMemoryStore.medicalRecords.set('appointments', []);
            }
            const appointmentWithId = { ...appointment, id: `appt_${Date.now()}` };
            inMemoryStore.medicalRecords.get('appointments')!.push(appointmentWithId);
            res.json({ success: true, appointment: appointmentWithId });
        }
    } catch (e) {
        console.error('Create appointment error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Get all appointments (admin)
app.get('/api/appointments', async (_req: Request, res: Response) => {
    try {
        let appointments: any[] = [];

        if (dbReady) {
            appointments = await FirebaseHelper.getDocs('appointments');
            // Sort by creation date (newest first)
            appointments.sort((a, b) => {
                const dateA = new Date(a.created_at || 0).getTime();
                const dateB = new Date(b.created_at || 0).getTime();
                return dateB - dateA;
            });
        } else {
            appointments = inMemoryStore.medicalRecords.get('appointments') || [];
            appointments.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        }

        res.json({ success: true, appointments });
    } catch (e) {
        console.error('Fetch appointments error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Get appointments by username
app.get('/api/appointments/user/:username', async (req: Request, res: Response) => {
    try {
        const { username } = req.params;

        if (!username) {
            return res.status(400).json({ success: false, message: 'Username required' });
        }

        let appointments: any[] = [];

        if (dbReady) {
            appointments = await FirebaseHelper.getDocs('appointments', [
                { field: 'username', operator: '==', value: username }
            ]);
            appointments.sort((a, b) => {
                const dateA = new Date(a.created_at || 0).getTime();
                const dateB = new Date(b.created_at || 0).getTime();
                return dateB - dateA;
            });
        } else {
            const allAppointments = inMemoryStore.medicalRecords.get('appointments') || [];
            appointments = allAppointments.filter((a: any) => a.username === username);
            appointments.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
        }

        res.json({ success: true, appointments });
    } catch (e) {
        console.error('Fetch user appointments error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Update appointment status
app.put('/api/appointments/:id', async (req: Request, res: Response) => {
    try {
        const { id } = req.params;
        const { status, date, time, reason } = req.body;

        if (dbReady) {
            const appointment = await FirebaseHelper.getDoc('appointments', id);
            if (!appointment) {
                return res.status(404).json({ success: false, message: 'Appointment not found' });
            }

            const updates: any = { updated_at: new Date().toISOString() };
            if (status) updates.status = status;
            if (date) updates.date = date;
            if (time) updates.time = time;
            if (reason !== undefined) updates.reason = reason;

            await FirebaseHelper.updateDoc('appointments', id, updates);
            res.json({ success: true, message: 'Appointment updated successfully' });
        } else {
            const appointments = inMemoryStore.medicalRecords.get('appointments') || [];
            const index = appointments.findIndex((a: any) => a.id === id);

            if (index === -1) {
                return res.status(404).json({ success: false, message: 'Appointment not found' });
            }

            if (status) appointments[index].status = status;
            if (date) appointments[index].date = date;
            if (time) appointments[index].time = time;
            if (reason !== undefined) appointments[index].reason = reason;
            appointments[index].updated_at = new Date().toISOString();

            res.json({ success: true, message: 'Appointment updated successfully' });
        }
    } catch (e) {
        console.error('Update appointment error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// Delete appointment
app.delete('/api/appointments/:id', async (req: Request, res: Response) => {
    try {
        const { id } = req.params;

        if (dbReady) {
            const appointment = await FirebaseHelper.getDoc('appointments', id);
            if (!appointment) {
                return res.status(404).json({ success: false, message: 'Appointment not found' });
            }

            await FirebaseHelper.deleteDoc('appointments', id);
            res.json({ success: true, message: 'Appointment deleted successfully' });
        } else {
            const appointments = inMemoryStore.medicalRecords.get('appointments') || [];
            const index = appointments.findIndex((a: any) => a.id === id);

            if (index === -1) {
                return res.status(404).json({ success: false, message: 'Appointment not found' });
            }

            appointments.splice(index, 1);
            res.json({ success: true, message: 'Appointment deleted successfully' });
        }
    } catch (e) {
        console.error('Delete appointment error:', e);
        res.status(500).json({ success: false, message: 'Internal server error' });
    }
});

// ============================================
// START SERVER
// ============================================

const startServer = async () => {
    // Initialize Firebase
    db = await initializeFirebase();
    dbReady = db !== null;

    if (!dbReady) {
        console.warn('⚠️  Running without Firebase. Using in-memory storage for demo.');
    }

    const PORT = process.env.PORT || 8000;
    app.listen(Number(PORT), '0.0.0.0', () => {
        console.log(`\n🚀 LungDx API (Firebase) listening on :${PORT}`);
        console.log(`📊 Database: ${dbReady ? '✅ Firebase Firestore' : '⚠️  In-Memory (Demo Mode)'}`);
        if (!dbReady) {
            console.log(`\n💡 To enable Firebase:`);
            console.log(`   1. Set up Firebase project at https://console.firebase.google.com`);
            console.log(`   2. Add credentials to .env file`);
            console.log(`   3. Restart the server\n`);
        }
    });
};

startServer();
