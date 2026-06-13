
import 'dotenv/config';
import { initializeFirebase, FirebaseHelper } from './src/firebase';

const encodeBase64 = (s: string) => Buffer.from(s, 'utf-8').toString('base64');

async function seedAdmin() {
    console.log('Initializing Firebase...');
    const db = await initializeFirebase();

    if (!db) {
        console.error('❌ Failed to initialize Firebase. Cannot seed admin.');
        process.exit(1);
    }

    console.log('Checking for admin user...');
    const adminUsername = 'admin';
    const existingAdmin = await FirebaseHelper.findOne('users', 'username', adminUsername);

    if (existingAdmin) {
        console.log('✅ Admin user already exists.');
        // Optional: Update password if needed, but for now just exit
        process.exit(0);
    }

    console.log('Creating admin user...');
    const adminUser = {
        username: adminUsername,
        firstName: 'System',
        lastName: 'Admin',
        email: 'admin@lungdx.com',
        password: encodeBase64('admin12345'),
        role: 'admin',
        created_at: new Date().toISOString(),
        loginAttempts: 0,
        profilePicture: '',
        phoneNumber: '',
        dateOfBirth: '',
        gender: 'Other'
    };

    await FirebaseHelper.createDoc('users', adminUser, adminUsername);
    console.log('✅ Admin user created successfully.');
    console.log('Username: admin');
    console.log('Password: admin12345');
    process.exit(0);
}

seedAdmin().catch(console.error);
