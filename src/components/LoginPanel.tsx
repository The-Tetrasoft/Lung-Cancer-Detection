import React, { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { Eye, EyeOff, User, Lock } from 'lucide-react';
import { AnimatedDoctor } from './AnimatedDoctor';
import { auth, db } from '../firebase-client';
import { signInWithEmailAndPassword } from 'firebase/auth';
import { collection, query, where, getDocs } from 'firebase/firestore';

interface LoginPanelProps {
  onLoginSuccess: (user: any) => void;
  onNavigateToSignup: () => void;
  onNavigateToAbout: () => void;
  onNavigateToContact: () => void;
}

export function LoginPanel({ onLoginSuccess, onNavigateToSignup, onNavigateToAbout, onNavigateToContact }: LoginPanelProps) {
  const [usernameOrEmail, setUsernameOrEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      let email = usernameOrEmail;

      // If input is not an email, try to find the email by username
      if (!usernameOrEmail.includes('@')) {
        try {
          const usersRef = collection(db, 'users');
          const q = query(usersRef, where('username', '==', usernameOrEmail));
          const querySnapshot = await getDocs(q);

          if (querySnapshot.empty) {
            throw new Error('Username not found');
          }

          const userData = querySnapshot.docs[0].data();
          email = userData.email;
        } catch (queryErr: any) {
          if (queryErr.code === 'permission-denied') {
            throw new Error('Database access denied. Please log in with Email instead of Username, or update Firestore Rules.');
          }
          throw queryErr;
        }
      }

      // Sign in with Firebase Auth
      const userCredential = await signInWithEmailAndPassword(auth, email, password);
      const user = userCredential.user;

      // Fetch user details from Firestore
      // Note: We need to find the user doc by email or uid. 
      // Ideally, we store user data in a doc with ID = uid.
      // But since we might have legacy data, let's query by email if needed.

      // Try fetching by UID first (best practice)
      // But for now, let's assume we need to query 'users' collection to get the role and other details
      // Fetch user details from Firestore
      const usersRef = collection(db, 'users');
      const q = query(usersRef, where('email', '==', email));
      let userData;

      try {
        const querySnapshot = await getDocs(q);
        if (!querySnapshot.empty) {
          userData = querySnapshot.docs[0].data();
        }
      } catch (firestoreErr) {
        console.warn('Firestore read failed, using Auth profile:', firestoreErr);
      }

      if (userData) {
        // Force admin role for specific email (Backdoor/Fix)
        // Force admin role for specific email (Backdoor/Fix)
        const adminEmails = ['admin@lungdx.com', 'admin@lungcancer.system'];
        if (adminEmails.includes(userData.email) && userData.role !== 'admin') {
          console.log(`Promoting ${userData.email} to admin role`);
          userData.role = 'admin';
          // Try to update Firestore to persist this
          try {
            const { updateDoc, doc } = await import('firebase/firestore');
            const usersRef = collection(db, 'users');
            const q = query(usersRef, where('email', '==', userData.email));
            const snap = await getDocs(q);
            if (!snap.empty) {
              await updateDoc(doc(db, 'users', snap.docs[0].id), { role: 'admin' });
            }
          } catch (e) {
            console.error("Failed to persist admin role update:", e);
          }
        }

        setSuccess('Login successful!');
        setTimeout(() => {
          onLoginSuccess(userData);
        }, 1000);
      } else {
        // Fallback: Use Auth data if Firestore profile is missing or inaccessible
        console.log('User profile not found in Firestore, using Auth data');
        const fallbackUser = {
          username: user.displayName || email?.split('@')[0] || 'User',
          firstName: user.displayName?.split(' ')[0] || 'User',
          lastName: user.displayName?.split(' ')[1] || '',
          email: user.email,
          role: 'user', // Default role
          uid: user.uid
        };

        // AUTO-REPAIR: Try to create the missing profile now!
        // This ensures that users who signed up when Rules were broken still get a DB entry.
        try {
          // We use addDoc to match the SignupPanel pattern
          // Note: We don't have phone/DOB here, but at least the user exists in DB.
          const { addDoc } = await import('firebase/firestore');
          await addDoc(collection(db, 'users'), {
            uid: user.uid,
            email: user.email,
            username: fallbackUser.username,
            firstName: fallbackUser.firstName,
            lastName: fallbackUser.lastName,
            role: 'user',
            createdAt: new Date().toISOString(),
            isRepairedProfile: true,
            phoneNumber: '', // Missing
            dateOfBirth: '', // Missing
            gender: '' // Missing
          });
          console.log("Auto-repaired missing Firestore profile");
        } catch (repairErr) {
          console.error("Failed to auto-repair profile:", repairErr);
        }

        setSuccess('Login successful! (Profile repaired)');
        setTimeout(() => {
          onLoginSuccess(fallbackUser);
        }, 1000);
      }

    } catch (err: any) {
      console.error('Login error:', err);
      let msg = 'Login failed. Please check your credentials.';

      // Handle specific Firebase Auth errors
      if (err.code === 'auth/invalid-credential' || err.code === 'auth/user-not-found' || err.code === 'auth/wrong-password') {
        msg = 'Invalid username/email or password.';
      } else if (err.message === 'Username not found') {
        msg = 'Username not found. Please sign up first.';
      } else if (err.code === 'auth/network-request-failed') {
        msg = 'Network error. Please check your internet connection.';
      } else if (err.code === 'auth/invalid-api-key') {
        msg = 'Configuration Error: Invalid Firebase API Key. Please check your .env file.';
      } else if (err.code === 'auth/configuration-not-found') {
        msg = 'Configuration Error: Authentication is not enabled in Firebase Console. Please enable "Email/Password" sign-in.';
      } else if (err.code === 'permission-denied') {
        msg = 'Access Denied: You do not have permission to access the database. Please log in with Email.';
      } else if (err.message) {
        // Show the actual error message for debugging
        msg = `Error: ${err.message}`;
      }

      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Logo/Header */}
        <div className="text-center">
          <div className="mx-auto w-20 h-20 mb-4 flex items-center justify-center">
            <AnimatedDoctor className="w-full h-full" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">LungDx😊</h1>
          <p className="text-slate-600 mt-2">Advanced AI-powered lung diagnosis</p>
        </div>

        {success && (
          <Alert className="border-green-200 bg-green-50">
            <AlertDescription className="text-green-800">{success}</AlertDescription>
          </Alert>
        )}

        {error && (
          <Alert className="border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        {/* Login Card */}
        <Card className="shadow-lg">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center">Sign In</CardTitle>
            <CardDescription className="text-center">
              Enter your credentials to access your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="usernameOrEmail">Username or Email</Label>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="usernameOrEmail"
                    type="text"
                    placeholder="Enter username or email"
                    value={usernameOrEmail}
                    onChange={(e) => setUsernameOrEmail(e.target.value)}
                    className="pl-10"
                    required
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="pl-10 pr-10"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={loading}
              >
                {loading ? 'Signing In...' : 'Sign In'}
              </Button>

              <div className="text-xs text-center text-slate-400 mt-2">
                Secure Login via Firebase
              </div>
            </form>

            <div className="mt-6">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-white px-2 text-gray-500">Don't have an account?</span>
                </div>
              </div>
              <Button
                type="button"
                variant="outline"
                className="w-full mt-4"
                onClick={onNavigateToSignup}
              >
                Create Account
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Footer Links */}
        <div className="flex justify-center space-x-6 text-sm">
          <button
            onClick={onNavigateToAbout}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            About Us
          </button>
          <button
            onClick={onNavigateToContact}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            Contact
          </button>
        </div>
      </div>
    </div>
  );
}