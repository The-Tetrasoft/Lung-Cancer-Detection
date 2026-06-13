import React from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { LogOut, Upload, FileText, User, Settings, Stethoscope } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';

interface UserDashboardProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
    profilePicture?: string;
  };
  onLogout: () => void;
  onNavigateToMedicalHistory: () => void;
  onNavigateToFindSpecialists: () => void;
  onNavigateToStartAnalysis: () => void;
  onNavigateToProfile: () => void;
}

export function UserDashboard({ user, onLogout, onNavigateToMedicalHistory, onNavigateToFindSpecialists, onNavigateToStartAnalysis, onNavigateToProfile }: UserDashboardProps) {
  return (
    <div>
      {/* Match static pages: center content and limit width */}
      <div className="max-w-6xl mx-auto px-4 space-y-6 pt-8">
        {/* Header */}
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">LungDx😊</h1>
            <p className="text-slate-600">Welcome back, {user.firstName} {user.lastName}</p>
          </div>
          <Button onClick={onLogout} variant="outline" className="flex items-center border-slate-200 hover:bg-white hover:text-blue-600">
            <LogOut className="w-4 h-4 mr-2" />
            Sign Out
          </Button>
        </div>

        {/* Account Information (top) */}
        <div className="">
          <Card className="mb-6 bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center text-slate-800">
                <User className="w-5 h-5 mr-2 text-blue-600" />
                Account Information
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-6 mb-6">
                <Avatar className="w-16 h-16 border-2 border-blue-200">
                  <AvatarImage src={user.profilePicture} />
                  <AvatarFallback className="text-lg bg-blue-100 text-blue-700">
                    {user.firstName.charAt(0)}{user.lastName.charAt(0)}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="text-lg font-semibold text-slate-900">
                    {user.firstName} {user.lastName}
                  </h3>
                  <p className="text-slate-600">@{user.username}</p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-slate-500">Username</p>
                  <p className="font-medium text-slate-900">{user.username}</p>
                </div>
                <div>
                  <p className="text-sm text-slate-500">Email</p>
                  <p className="font-medium text-slate-900">{user.email}</p>
                </div>
                <div>
                  <p className="text-sm text-slate-500">Account Type</p>
                  <p className="font-medium capitalize text-slate-900">{user.role}</p>
                </div>
              </div>
            </CardContent>
          </Card>

        </div>

        {/* Main Features (centered at top) */}
        <div className="mt-6">
          <div className="w-full grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card
              className="hover:shadow-xl transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 bg-white/80 backdrop-blur-sm border-blue-100 group"
              onClick={onNavigateToStartAnalysis}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigateToStartAnalysis(); } }}
            >
              <CardHeader className="text-center">
                <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Upload className="w-8 h-8 text-blue-600" />
                </div>
                <CardTitle className="text-slate-800">Upload X-Ray</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-600 text-center mb-4 text-sm">
                  Upload chest X-ray images for AI-powered lung diagnosis analysis
                </p>
                <Button className="w-full bg-blue-600 hover:bg-blue-700">
                  Start Analysis
                </Button>
              </CardContent>
            </Card>

            <Card
              className="hover:shadow-xl transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-400 bg-white/80 backdrop-blur-sm border-teal-100 group"
              onClick={onNavigateToMedicalHistory}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigateToMedicalHistory(); } }}
            >
              <CardHeader className="text-center">
                <div className="w-16 h-16 mx-auto bg-teal-100 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <FileText className="w-8 h-8 text-teal-600" />
                </div>
                <CardTitle className="text-slate-800">Medical History</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-600 text-center mb-4 text-sm">
                  View your previous analysis results and track your medical history
                </p>
                <Button variant="outline" className="w-full border-teal-200 text-teal-700 hover:bg-teal-50">
                  View History
                </Button>
              </CardContent>
            </Card>

            <Card
              className="hover:shadow-xl transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 bg-white/80 backdrop-blur-sm border-indigo-100 group"
              onClick={onNavigateToFindSpecialists}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigateToFindSpecialists(); } }}
            >
              <CardHeader className="text-center">
                <div className="w-16 h-16 mx-auto bg-indigo-100 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Stethoscope className="w-8 h-8 text-indigo-600" />
                </div>
                <CardTitle className="text-slate-800">Find Specialists</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-600 text-center mb-4 text-sm">
                  Connect with qualified pulmonologists and oncologists for consultation
                </p>
                <Button variant="outline" className="w-full border-indigo-200 text-indigo-700 hover:bg-indigo-50">
                  Find Doctors
                </Button>
              </CardContent>
            </Card>

            <Card
              className="hover:shadow-xl transition-all cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 bg-white/80 backdrop-blur-sm border-purple-100 group"
              onClick={onNavigateToProfile}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onNavigateToProfile(); } }}
            >
              <CardHeader className="text-center">
                <div className="w-16 h-16 mx-auto bg-purple-100 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                  <Settings className="w-8 h-8 text-purple-600" />
                </div>
                <CardTitle className="text-slate-800">Account Settings</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-600 text-center mb-4 text-sm">
                  Manage your account preferences and update your personal information
                </p>
                {/* Button is purely visual here, clicks pass through to the Card */}
                <div className="w-full">
                  <Button
                    variant="outline"
                    className="w-full border-purple-200 text-purple-700 hover:bg-purple-50 pointer-events-none"
                    tabIndex={-1} // Remove from tab order since the card handles focus
                  >
                    Settings
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* (removed duplicate user info card which is now above) */}

        {/* Information Section */}
        <Card className="bg-white/90 backdrop-blur-sm border-slate-200 shadow-md">
          <CardHeader>
            <CardTitle className="text-slate-800">How It Works</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-4xl mb-4">📤</div>
                <h3 className="font-semibold mb-2 text-slate-900">1. Upload Image</h3>
                <p className="text-slate-600 text-sm">Upload your chest X-ray or CT scan image in supported formats</p>
              </div>
              <div className="text-center">
                <div className="text-4xl mb-4">🤖</div>
                <h3 className="font-semibold mb-2 text-slate-900">2. AI Analysis</h3>
                <p className="text-slate-600 text-sm">Our advanced AI model analyzes the image for potential lung cancer indicators</p>
              </div>
              <div className="text-center">
                <div className="text-4xl mb-4">📊</div>
                <h3 className="font-semibold mb-2 text-slate-900">3. Get Results</h3>
                <p className="text-slate-600 text-sm">Receive detailed analysis results with confidence scores and recommendations</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Disclaimer */}
        <Card className="bg-amber-50 border-amber-200 shadow-sm">
          <CardContent className="pt-6">
            <div className="flex items-start space-x-3">
              <div className="text-2xl">⚠️</div>
              <div>
                <h3 className="font-semibold text-amber-900">Medical Disclaimer</h3>
                <p className="text-amber-800 mt-1 text-sm">
                  This AI tool is designed to assist healthcare professionals and should not replace professional medical diagnosis.
                  Always consult with qualified healthcare providers for medical advice and treatment decisions.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}