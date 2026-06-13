import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Progress } from './ui/progress';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Eye, EyeOff, User, Mail, Phone, Lock, Calendar, ArrowLeft, Upload } from 'lucide-react';
import { auth, db } from '../firebase-client';
import { createUserWithEmailAndPassword, updateProfile } from 'firebase/auth';
import { collection, addDoc, query, where, getDocs } from 'firebase/firestore';

interface SignupPanelProps {
  onSignupSuccess: () => void;
  onNavigateToLogin: () => void;
  onNavigateToTerms: () => void;
}

interface FormData {
  username: string;
  firstName: string;
  lastName: string;
  dateOfBirth: string;
  gender: string;
  email: string;
  phoneNumber: string;
  password: string;
  confirmPassword: string;
  agreeToTerms: boolean;
  profilePictureFile: File | null;
  profilePicturePreview: string;
}

interface FormErrors {
  [key: string]: string;
}

export function SignupPanel({ onSignupSuccess, onNavigateToLogin, onNavigateToTerms }: SignupPanelProps) {
  const [formData, setFormData] = useState<FormData>({
    username: '',
    firstName: '',
    lastName: '',
    dateOfBirth: '',
    gender: '',
    email: '',
    phoneNumber: '',
    password: '',
    confirmPassword: '',
    agreeToTerms: false,
    profilePictureFile: null,
    profilePicturePreview: ''
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState(0);
  const [passwordStrengthText, setPasswordStrengthText] = useState('');
  const [isFormValid, setIsFormValid] = useState(false);

  // Password strength calculation
  const calculatePasswordStrength = (password: string) => {
    let score = 0;
    if (password.length >= 6) score++;
    if (/[a-z]/.test(password)) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[!@#$%^&*(),.?":{}|<>]/.test(password)) score++;

    setPasswordStrength(score);

    if (score < 3) {
      setPasswordStrengthText('Weak password, enter a strong password');
    } else if (score === 3) {
      setPasswordStrengthText('Medium strength');
    } else if (score === 4) {
      setPasswordStrengthText('Strong password');
    } else {
      setPasswordStrengthText('Very strong password');
    }
  };

  // Profile picture upload handler
  const handleProfilePictureUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setErrors({ ...errors, profilePicture: 'Only JPEG, PNG, and WebP images are allowed' });
      return;
    }

    // Validate file size (max 5MB)
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
      setErrors({ ...errors, profilePicture: 'Image size must be less than 5MB' });
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setFormData(prev => ({
        ...prev,
        profilePictureFile: file,
        profilePicturePreview: e.target?.result as string
      }));
      setErrors(prev => ({ ...prev, profilePicture: '' }));
    };
    reader.readAsDataURL(file);
  };

  // Validation functions
  const validateField = async (name: string, value: string) => {
    const newErrors = { ...errors };

    switch (name) {
      case 'username':
        if (!value) {
          newErrors.username = 'Username is required';
        } else if (value.length < 3) {
          newErrors.username = 'Username must be at least 3 characters';
        } else {
          // Valid length, so clear the error immediately
          delete newErrors.username;

          // Check username availability in Firestore
          try {
            const usersRef = collection(db, 'users');
            const q = query(usersRef, where('username', '==', value));
            const querySnapshot = await getDocs(q);

            if (!querySnapshot.empty) {
              newErrors.username = 'Username already exists, please choose another';
            }
          } catch (error) {
            console.error('Username check error:', error);
            // If check fails, we assume valid to not block user, or we could show a warning.
            // But definitely don't show "must be 3 characters".
          }
        }
        break;

      case 'firstName':
        if (!value) {
          newErrors.firstName = 'First name is required';
        } else {
          delete newErrors.firstName;
        }
        break;

      case 'lastName':
        if (!value) {
          newErrors.lastName = 'Last name is required';
        } else {
          delete newErrors.lastName;
        }
        break;

      case 'dateOfBirth':
        if (!value) {
          newErrors.dateOfBirth = 'Date of birth is required';
        } else {
          const birthDate = new Date(value);
          const today = new Date();
          const age = today.getFullYear() - birthDate.getFullYear();
          if (age < 18) {
            newErrors.dateOfBirth = 'You must be at least 18 years old';
          } else {
            delete newErrors.dateOfBirth;
          }
        }
        break;

      case 'gender':
        if (!value) {
          newErrors.gender = 'Gender is required';
        } else {
          delete newErrors.gender;
        }
        break;

      case 'email':
        if (!value) {
          newErrors.email = 'Email is required';
        } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
          newErrors.email = 'Please enter a valid email address';
        } else {
          delete newErrors.email;
        }
        break;

      case 'phoneNumber':
        if (!value) {
          newErrors.phoneNumber = 'Phone number is required';
        } else if (!/^\d{11}$/.test(value)) {
          newErrors.phoneNumber = 'Phone number must be exactly 11 digits';
        } else {
          delete newErrors.phoneNumber;
        }
        break;

      case 'password':
        if (!value) {
          newErrors.password = 'Password is required';
        } else if (passwordStrength < 3) {
          newErrors.password = 'Password is too weak';
        } else {
          delete newErrors.password;
        }
        break;

      case 'confirmPassword':
        if (!value) {
          newErrors.confirmPassword = 'Please confirm your password';
        } else if (value !== formData.password) {
          newErrors.confirmPassword = 'Passwords do not match';
        } else {
          delete newErrors.confirmPassword;
        }
        break;
    }

    setErrors(newErrors);
  };

  // Handle input changes
  const handleInputChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));

    if (name === 'password') {
      calculatePasswordStrength(value);
    }

    validateField(name, value);
  };

  // Check if form is valid
  useEffect(() => {
    const requiredFields = ['username', 'firstName', 'lastName', 'dateOfBirth', 'gender', 'email', 'phoneNumber', 'password', 'confirmPassword'];
    const allFieldsFilled = requiredFields.every(field => formData[field as keyof FormData]);

    // Filter out empty profilePicture errors (since it's optional)
    const relevantErrors = Object.entries(errors).filter(([key, value]) => {
      if (key === 'profilePicture' && !value) return false;
      return value && typeof value === 'string' && value.trim() !== '';
    });
    const noErrors = relevantErrors.length === 0;

    const strongPassword = passwordStrength >= 3;
    const termsAccepted = formData.agreeToTerms;

    setIsFormValid(allFieldsFilled && noErrors && strongPassword && termsAccepted);
  }, [formData, errors, passwordStrength]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!isFormValid) return;

    setLoading(true);

    try {
      // 1. Create Auth User
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      const user = userCredential.user;

      // 2. Update Profile (Display Name)
      await updateProfile(user, {
        displayName: `${formData.firstName} ${formData.lastName}`
      });

      // 3. Create User Document in Firestore
      try {
        await addDoc(collection(db, 'users'), {
          uid: user.uid,
          username: formData.username,
          firstName: formData.firstName,
          lastName: formData.lastName,
          dateOfBirth: formData.dateOfBirth,
          gender: formData.gender,
          email: formData.email,
          phoneNumber: formData.phoneNumber,
          role: formData.email.toLowerCase() === 'admin@lungdx.com' ? 'admin' : 'user',
          profilePicture: formData.profilePicturePreview || '', // Storing base64 for now (consider Storage later)
          createdAt: new Date().toISOString()
        });
      } catch (firestoreError) {
        console.error("Error creating user profile in Firestore:", firestoreError);
        // We continue to success even if Firestore fails, because the Auth account IS created.
        // The LoginPanel has fallback logic to handle missing Firestore profiles.
      }

      onSignupSuccess();

    } catch (error: any) {
      console.error('Signup error:', error);
      let msg = 'Signup failed. Please try again.';

      if (error.code === 'auth/email-already-in-use') {
        msg = 'Email is already in use. Please log in instead.';
      } else if (error.code === 'auth/weak-password') {
        msg = 'Password is too weak. It should be at least 6 characters.';
      } else if (error.code === 'auth/operation-not-allowed') {
        msg = 'Configuration Error: Email/Password sign-in is not enabled in Firebase Console.';
      } else if (error.code === 'auth/invalid-email') {
        msg = 'Invalid email address.';
      } else if (error.code === 'permission-denied') {
        msg = 'Database Access Error: Permission denied. Please check Firestore Rules.';
      } else if (error.message) {
        msg = `Error: ${error.message}`;
      }

      setErrors({ submit: msg });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 flex items-center justify-center p-4">
      <div className="w-full max-w-2xl space-y-6">
        {/* Header */}
        <div className="text-center">
          <button
            onClick={onNavigateToLogin}
            className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Login
          </button>
          <h1 className="text-3xl font-bold text-slate-900">Create Account</h1>
          <p className="text-slate-600 mt-2">Join our lung diagnosis platform</p>
        </div>

        {/* Signup Card */}
        <Card className="shadow-xl border-blue-100 bg-white/90 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-2xl text-center text-slate-800">Sign Up</CardTitle>
            <CardDescription className="text-center text-slate-600">
              Fill in the form below to create your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Username */}
                <div className="space-y-2">
                  <Label htmlFor="username">Username *</Label>
                  <div className="relative">
                    <User className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                    <Input
                      id="username"
                      type="text"
                      placeholder="Choose a username"
                      value={formData.username}
                      onChange={(e) => handleInputChange('username', e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  {errors.username && <p className="text-sm text-red-600">{errors.username}</p>}
                </div>

                {/* First Name */}
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name *</Label>
                  <Input
                    id="firstName"
                    type="text"
                    placeholder="Enter first name"
                    value={formData.firstName}
                    onChange={(e) => handleInputChange('firstName', e.target.value)}
                  />
                  {errors.firstName && <p className="text-sm text-red-600">{errors.firstName}</p>}
                </div>

                {/* Last Name */}
                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name *</Label>
                  <Input
                    id="lastName"
                    type="text"
                    placeholder="Enter last name"
                    value={formData.lastName}
                    onChange={(e) => handleInputChange('lastName', e.target.value)}
                  />
                  {errors.lastName && <p className="text-sm text-red-600">{errors.lastName}</p>}
                </div>

                {/* Date of Birth */}
                <div className="space-y-2">
                  <Label htmlFor="dateOfBirth">Date of Birth *</Label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                    <Input
                      id="dateOfBirth"
                      type="date"
                      value={formData.dateOfBirth}
                      onChange={(e) => handleInputChange('dateOfBirth', e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  {errors.dateOfBirth && <p className="text-sm text-red-600">{errors.dateOfBirth}</p>}
                </div>

                {/* Gender */}
                <div className="space-y-2">
                  <Label htmlFor="gender">Gender *</Label>
                  <Select value={formData.gender} onValueChange={(value) => handleInputChange('gender', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.gender && <p className="text-sm text-red-600">{errors.gender}</p>}
                </div>

                {/* Email */}
                <div className="space-y-2">
                  <Label htmlFor="email">Email *</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter email address"
                      value={formData.email}
                      onChange={(e) => handleInputChange('email', e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  {errors.email && <p className="text-sm text-red-600">{errors.email}</p>}
                </div>
              </div>

              {/* Phone Number */}
              <div className="space-y-2">
                <Label htmlFor="phoneNumber">Phone Number *</Label>
                <div className="relative">
                  <Phone className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    id="phoneNumber"
                    type="tel"
                    placeholder="Enter 11-digit phone number"
                    value={formData.phoneNumber}
                    onChange={(e) => handleInputChange('phoneNumber', e.target.value)}
                    className="pl-10"
                  />
                </div>
                {errors.phoneNumber && <p className="text-sm text-red-600">{errors.phoneNumber}</p>}
              </div>

              {/* Profile Picture (Optional) */}
              <div className="space-y-2">
                <Label>Profile Picture (Optional)</Label>
                <div className="flex items-center gap-4">
                  <Avatar className="w-16 h-16 border-2 border-blue-100">
                    <AvatarImage src={formData.profilePicturePreview} />
                    <AvatarFallback className="bg-blue-50 text-blue-600">
                      <User className="w-6 h-6" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <Label htmlFor="profile-picture-upload" className="cursor-pointer">
                      <div className="border-2 border-dashed border-blue-200 rounded-lg p-4 text-center hover:border-blue-400 hover:bg-blue-50 transition-colors">
                        <Upload className="w-6 h-6 mx-auto text-blue-400 mb-2" />
                        <p className="text-sm text-slate-600">
                          Click to upload profile picture
                        </p>
                        <p className="text-xs text-slate-500 mt-1">
                          JPEG, PNG, WebP up to 5MB
                        </p>
                      </div>
                    </Label>
                    <input
                      id="profile-picture-upload"
                      type="file"
                      accept="image/jpeg,image/jpg,image/png,image/webp"
                      onChange={handleProfilePictureUpload}
                      className="hidden"
                    />
                    {errors.profilePicture && (
                      <p className="text-red-600 text-sm mt-2">{errors.profilePicture}</p>
                    )}
                  </div>
                </div>
              </div>

              {/* Password */}
              <div className="space-y-2">
                <Label htmlFor="password">Password *</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter password"
                    value={formData.password}
                    onChange={(e) => handleInputChange('password', e.target.value)}
                    className="pl-10 pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-3 text-slate-400 hover:text-slate-600"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {formData.password && (
                  <div className="space-y-2">
                    <Progress value={(passwordStrength / 5) * 100} className="h-2" />
                    <p className={`text-sm ${passwordStrength < 3 ? 'text-red-600' : 'text-green-600'}`}>
                      {passwordStrengthText}
                    </p>
                  </div>
                )}
                {errors.password && <p className="text-sm text-red-600">{errors.password}</p>}
              </div>

              {/* Confirm Password */}
              <div className="space-y-2">
                <Label htmlFor="confirmPassword">Confirm Password *</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? "text" : "password"}
                    placeholder="Confirm password"
                    value={formData.confirmPassword}
                    onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                    className="pl-10 pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-3 text-slate-400 hover:text-slate-600"
                  >
                    {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                {errors.confirmPassword && <p className="text-sm text-red-600">{errors.confirmPassword}</p>}
              </div>

              {/* Terms and Conditions */}
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="agreeToTerms"
                  checked={formData.agreeToTerms}
                  onCheckedChange={(checked) => setFormData(prev => ({ ...prev, agreeToTerms: checked as boolean }))}
                />
                <Label htmlFor="agreeToTerms" className="text-sm text-slate-700">
                  I agree to the{' '}
                  <button
                    type="button"
                    onClick={onNavigateToTerms}
                    className="text-blue-600 hover:text-blue-800 underline"
                  >
                    Terms and Conditions
                  </button>
                </Label>
              </div>

              {errors.submit && (
                <Alert className="border-red-200 bg-red-50">
                  <AlertDescription className="text-red-800">{errors.submit}</AlertDescription>
                </Alert>
              )}

              {!isFormValid && Object.keys(errors).length > 0 && (
                <p className="text-sm text-red-600 text-center">
                  Please fill up the form correctly
                </p>
              )}

              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700"
                disabled={!isFormValid || loading}
              >
                {loading ? 'Creating Account...' : 'Create Account'}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Terms and Conditions Link */}
        <div className="text-center">
          <button
            onClick={onNavigateToTerms}
            className="text-blue-600 hover:text-blue-800 hover:underline"
          >
            Terms and Conditions
          </button>
        </div>
      </div>
    </div>
  );
}