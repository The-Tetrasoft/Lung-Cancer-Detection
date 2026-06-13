import React, { useState, useEffect } from "react";
import { LoginPanel } from "./components/LoginPanel";
import { SignupPanel } from "./components/SignupPanel";
import { StaticPage } from "./components/StaticPage";
import { UserDashboard } from "./components/UserDashboard";
import { AdminPanel } from "./components/AdminPanel";
import { MedicalHistory } from "./components/MedicalHistory";
import { DoctorsList } from "./components/DoctorsList";
import { DoctorDetails } from "./components/DoctorDetails";
import { ManageDoctors } from "./components/ManageDoctors";
import { ManageUsers } from "./components/ManageUsers";
import { Analytics } from "./components/Analytics";
import { AddDoctorForm } from "./components/AddDoctorForm";
import { StartAnalysis } from "./components/StartAnalysis";
import { Profile } from "./components/Profile";
import { ManageAppointments } from "./components/ManageAppointments";
import { Alert, AlertDescription } from "./components/ui/alert";
import { auth, db } from "./firebase-client";
import { onAuthStateChanged } from "firebase/auth";
import { collection, query, where, getDocs } from "firebase/firestore";

type View =
  | "login"
  | "signup"
  | "aboutUs"
  | "contact"
  | "terms"
  | "userDashboard"
  | "adminPanel"
  | "medicalHistory"
  | "findSpecialists"
  | "doctorDetails"
  | "manageDoctors"
  | "manageUsers"
  | "analytics"
  | "addDoctor"
  | "startAnalysis"
  | "profile"
  | "manageAppointments";

interface User {
  username: string;
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  profilePicture?: string;
  phoneNumber?: string;
  dateOfBirth?: string; // ISO date string
  gender?: 'male' | 'female' | 'other';
}

export default function App() {
  const [currentView, setCurrentView] = useState<View>("login");
  const [user, setUser] = useState<User | null>(null);
  const [successMessage, setSuccessMessage] = useState("");
  const [history, setHistory] = useState<View[]>(["login"]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [isMobile, setIsMobile] = useState(false);
  const [selectedDoctorId, setSelectedDoctorId] = useState<string | null>(null);
  const [selectedSpecialty, setSelectedSpecialty] = useState<string | null>(null);
  const [editingDoctorId, setEditingDoctorId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Auth Listener
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        try {
          // Fetch user profile from Firestore
          const usersRef = collection(db, 'users');
          const q = query(usersRef, where('email', '==', currentUser.email));
          const querySnapshot = await getDocs(q);

          if (!querySnapshot.empty) {
            const userData = querySnapshot.docs[0].data() as User;

            // Force admin role for specific emails
            const adminEmails = ['admin@lungdx.com', 'admin@lungcancer.system'];
            if (adminEmails.includes(userData.email) && userData.role !== 'admin') {
              console.log(`Promoting ${userData.email} to admin role (App.tsx)`);
              userData.role = 'admin';
            }

            setUser(userData);

            // If on login page, redirect to dashboard
            if (currentView === 'login') {
              const targetView = userData.role === "admin" ? "adminPanel" : "userDashboard";
              setCurrentView(targetView);
              setHistory(['login', targetView]);
              setHistoryIndex(1);
            }
          } else {
            // Fallback: User exists in Auth but not in Firestore (or query failed empty)
            console.warn("User profile missing in Firestore, using Auth data fallback.");
            const fallbackUser: User = {
              username: currentUser.displayName || currentUser.email?.split('@')[0] || 'User',
              firstName: currentUser.displayName?.split(' ')[0] || 'User',
              lastName: currentUser.displayName?.split(' ')[1] || '',
              email: currentUser.email || '',
              role: 'user', // Default role
            };
            setUser(fallbackUser);

            if (currentView === 'login') {
              setCurrentView('userDashboard');
              setHistory(['login', 'userDashboard']);
              setHistoryIndex(1);
            }
          }
        } catch (error) {
          console.error("Error fetching user profile:", error);
          // Even if Firestore fails, let the user in if they are authenticated
          const fallbackUser: User = {
            username: currentUser.displayName || currentUser.email?.split('@')[0] || 'User',
            firstName: currentUser.displayName?.split(' ')[0] || 'User',
            lastName: currentUser.displayName?.split(' ')[1] || '',
            email: currentUser.email || '',
            role: 'user',
          };
          setUser(fallbackUser);

          if (currentView === 'login') {
            setCurrentView('userDashboard');
            setHistory(['login', 'userDashboard']);
            setHistoryIndex(1);
          }
        }
      } else {
        setUser(null);
        // If we are not on public pages, redirect to login
        if (!['login', 'signup', 'aboutUs', 'contact', 'terms'].includes(currentView)) {
          setCurrentView('login');
          setHistory(['login']);
          setHistoryIndex(0);
        }
      }
      setLoading(false);
    });

    // ensure viewport meta exists (for safe-area & mobile behavior)
    const existingViewport = document.querySelector('meta[name="viewport"]');
    if (!existingViewport) {
      const m = document.createElement('meta');
      m.name = 'viewport';
      m.content = 'width=device-width, initial-scale=1, viewport-fit=cover';
      document.head.appendChild(m);
    }

    // apple status bar style
    const appleStatusBar = document.querySelector('meta[name="apple-mobile-web-app-status-bar-style"]');
    if (!appleStatusBar) {
      const meta = document.createElement('meta');
      meta.name = 'apple-mobile-web-app-status-bar-style';
      meta.content = 'black-translucent';
      document.head.appendChild(meta);
    }

    // Mobile detection - use matchMedia for better accuracy
    const checkMobile = () => {
      // Check for mobile width OR if it's a touch device with coarse pointer (typical for mobile)
      const isMobileWidth = window.innerWidth < 1024;
      const isTouch = window.matchMedia('(pointer: coarse)').matches;

      // If it's narrow, it's definitely mobile view
      // If it's wider but touch (like a tablet), we might want desktop view but with touch adjustments
      // For now, stick to width-based for layout switching
      setIsMobile(isMobileWidth);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    window.addEventListener('orientationchange', checkMobile);

    // Prevent zoom on double tap (iOS Safari)
    let lastTouchEnd = 0;
    const preventZoom = (e: TouchEvent) => {
      const now = Date.now();
      if (now - lastTouchEnd <= 300) e.preventDefault();
      lastTouchEnd = now;
    };
    document.addEventListener('touchend', preventZoom, { passive: false });

    return () => {
      unsubscribe(); // Unsubscribe from auth listener
      window.removeEventListener('resize', checkMobile);
      window.removeEventListener('orientationchange', checkMobile);
      document.removeEventListener('touchend', preventZoom);
    };
  }, []);

  const handleLoginSuccess = (userData: User) => {
    setUser(userData);
    const targetView =
      userData.role === "admin"
        ? "adminPanel"
        : "userDashboard";
    handleNavigate(targetView);
  };

  const handleSignupSuccess = () => {
    setSuccessMessage(
      "Account created successfully! Please log in.",
    );
    handleNavigate("login");
    // Clear success message after 5 seconds
    setTimeout(() => setSuccessMessage(""), 5000);
  };

  const handleLogout = () => {
    auth.signOut().then(() => {
      setUser(null);
      // Reset history when logging out
      setHistory(["login"]);
      setHistoryIndex(0);
      setCurrentView("login");
    });
  };

  const handleNavigate = (view: View) => {
    // Add to history when navigating to a new view
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(view);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
    setCurrentView(view);
  };

  const handleGoBack = () => {
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setCurrentView(history[newIndex]);
    }
  };

  const handleGoForward = () => {
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      setHistoryIndex(newIndex);
      setCurrentView(history[newIndex]);
    }
  };

  const canGoBack = historyIndex > 0;
  const canGoForward = historyIndex < history.length - 1;

  const renderCurrentView = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center min-h-[50vh]">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      );
    }

    switch (currentView) {
      case "login":
        return (
          <>
            {successMessage && (
              <div className="fixed top-4 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-md">
                <Alert className="border-green-200 bg-green-50">
                  <AlertDescription className="text-green-800">
                    {successMessage}
                  </AlertDescription>
                </Alert>
              </div>
            )}
            <LoginPanel
              onLoginSuccess={handleLoginSuccess}
              onNavigateToSignup={() =>
                handleNavigate("signup")
              }
              onNavigateToAbout={() =>
                handleNavigate("aboutUs")
              }
              onNavigateToContact={() =>
                handleNavigate("contact")
              }
            />
          </>
        );

      case "signup":
        return (
          <SignupPanel
            onSignupSuccess={handleSignupSuccess}
            onNavigateToLogin={() => handleNavigate("login")}
            onNavigateToTerms={() => handleNavigate("terms")}
          />
        );

      case "aboutUs":
        return (
          <StaticPage
            pageType="aboutUs"
            onNavigateBack={() => handleGoBack()}
          />
        );

      case "contact":
        return (
          <StaticPage
            pageType="contact"
            onNavigateBack={() => handleGoBack()}
          />
        );

      case "terms":
        return (
          <StaticPage
            pageType="terms"
            onNavigateBack={() => handleGoBack()}
          />
        );

      case "userDashboard":
        return user ? (
          <UserDashboard
            user={user}
            onLogout={handleLogout}
            onNavigateToMedicalHistory={() =>
              handleNavigate("medicalHistory")
            }
            onNavigateToFindSpecialists={() =>
              handleNavigate("findSpecialists")
            }
            onNavigateToStartAnalysis={() =>
              handleNavigate("startAnalysis")
            }
            onNavigateToProfile={() =>
              handleNavigate("profile")
            }
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "startAnalysis":
        return user ? (
          <StartAnalysis
            user={user}
            onNavigateBack={() => handleGoBack()}
            onNavigateToDoctor={(doctorId) => {
              setSelectedDoctorId(doctorId);
              handleNavigate("doctorDetails");
            }}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "medicalHistory":
        return user ? (
          <MedicalHistory
            user={user}
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "profile":
        return user ? (
          <Profile
            user={user}
            onUserUpdate={(u) => setUser(prev => prev ? ({
              ...prev,
              ...u,
              profilePicture: u.profilePicture || undefined,
              phoneNumber: u.phoneNumber || undefined,
              dateOfBirth: u.dateOfBirth || undefined,
              gender: (u.gender || undefined) as any
            }) : prev)}
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() => handleNavigate("contact")}
          />
        );

      case "findSpecialists":
        return user ? (
          <DoctorsList
            user={user}
            selectedSpecialty={selectedSpecialty}
            onDoctorSelect={(doctorId) => {
              setSelectedDoctorId(doctorId);
              handleNavigate("doctorDetails");
            }}
            onSpecialtyFilter={(specialty) =>
              setSelectedSpecialty(specialty)
            }
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "doctorDetails":
        return user && selectedDoctorId ? (
          <DoctorDetails
            user={user}
            doctorId={selectedDoctorId}
            onNavigateBack={() => handleGoBack()}
            onSpecialtyFilter={(specialty) => {
              setSelectedSpecialty(specialty);
              handleNavigate("findSpecialists");
            }}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "adminPanel":
        return user && user.role === "admin" ? (
          <AdminPanel
            user={user}
            onLogout={handleLogout}
            onNavigateToManageDoctors={() =>
              handleNavigate("manageDoctors")
            }
            onNavigateToManageUsers={() =>
              handleNavigate("manageUsers")
            }
            onNavigateToAnalytics={() =>
              handleNavigate("analytics")
            }
            onNavigateToAppointments={() =>
              handleNavigate("manageAppointments")
            }
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "manageDoctors":
        return user && user.role === "admin" ? (
          <ManageDoctors
            onAddDoctor={() => {
              setEditingDoctorId(null);
              handleNavigate("addDoctor");
            }}
            onEditDoctor={(doctorId) => {
              setEditingDoctorId(doctorId);
              handleNavigate("addDoctor");
            }}
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "addDoctor":
        return user && user.role === "admin" ? (
          <AddDoctorForm
            doctorId={editingDoctorId}
            onSuccess={() => handleNavigate("manageDoctors")}
            onCancel={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "manageUsers":
        return user && user.role === "admin" ? (
          <ManageUsers
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "analytics":
        return user && user.role === "admin" ? (
          <Analytics
            onNavigateBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      case "manageAppointments":
        return user && user.role === "admin" ? (
          <ManageAppointments
            onBack={() => handleGoBack()}
          />
        ) : (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );

      default:
        return (
          <LoginPanel
            onLoginSuccess={handleLoginSuccess}
            onNavigateToSignup={() => handleNavigate("signup")}
            onNavigateToAbout={() => handleNavigate("aboutUs")}
            onNavigateToContact={() =>
              handleNavigate("contact")
            }
          />
        );
    }
  };

  return (
    <div className={`min-h-screen ${isMobile ? "mobile-app" : "desktop-app"} touch-manipulation flex`}>
      {/* Main wrapper */}
      <div
        className="flex-1 min-h-screen flex flex-col transition-all duration-300 ease-out"
      >
        {/* Fixed header */}
        <header className="fixed top-0 left-0 right-0 z-30 bg-white/80 backdrop-blur-sm border-b border-blue-100 shadow-sm" style={{ height: 'var(--header-height)', paddingTop: 'var(--safe-top)' }}>
          <div className="max-w-screen-xl mx-auto outer-margins px-6 h-full flex items-center justify-between">
            {/* Brand Title */}
            <div className="font-bold text-lg bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
              LungDx
            </div>
            <div></div>
          </div>
        </header>

        {/* Main content area */}
        <main
          className={`flex-1 ${isMobile ? "pb-safe-area-inset-bottom" : ""} bg-gradient-to-br from-blue-600 via-blue-500 to-cyan-400 min-h-screen`}
          style={{
            paddingTop: 'calc(var(--header-height) + 8px)',
          }}
        >
          <div className="w-full max-w-7xl mx-auto outer-margins px-4 py-6 sections-stack" style={{ minHeight: 'calc(100vh - var(--header-height))' }}>
            {renderCurrentView()}
          </div>
        </main>
      </div>
    </div>
  );
}