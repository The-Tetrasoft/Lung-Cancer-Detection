import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Users, Calendar, TrendingUp, Stethoscope, LogOut } from 'lucide-react';
import { db } from '../firebase-client';
import { collection, getDocs } from 'firebase/firestore';

interface AdminPanelProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
  };
  onLogout: () => void;
  onNavigateToManageDoctors: () => void;
  onNavigateToManageUsers: () => void;
  onNavigateToAnalytics: () => void;
  onNavigateToAppointments: () => void;
}

export function AdminPanel({
  user,
  onLogout,
  onNavigateToManageDoctors,
  onNavigateToManageUsers,
  onNavigateToAnalytics,
  onNavigateToAppointments
}: AdminPanelProps) {
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalDoctors: 0,
    pendingAppointments: 0,
    monthlyReports: 'Available'
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Fetch Users
        const usersSnap = await getDocs(collection(db, 'users'));
        const userCount = usersSnap.size;

        // Fetch Doctors
        const doctorsSnap = await getDocs(collection(db, 'doctors'));
        const doctorCount = doctorsSnap.size;

        // Fetch Appointments
        // Note: 'appointments' collection might not exist yet if no one booked.
        // We'll handle it gracefully.
        let apptCount = 0;
        try {
          const apptsSnap = await getDocs(collection(db, 'appointments'));
          apptCount = apptsSnap.size;
        } catch (e) {
          console.log("Appointments collection might be empty or missing", e);
        }

        setStats({
          totalUsers: userCount,
          totalDoctors: doctorCount,
          pendingAppointments: apptCount,
          monthlyReports: 'Available'
        });
      } catch (error) {
        console.error("Error fetching admin stats:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const cardClass = "bg-white/20 backdrop-blur-md border-2 border-white/40 shadow-xl hover:bg-white/30 hover:scale-[1.02] transition-all duration-300 cursor-pointer group rounded-xl";

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold text-white">Admin Dashboard</h1>
            <span className="px-4 py-1.5 bg-blue-500/30 text-white text-xs font-bold rounded-full backdrop-blur-sm border border-white/20 shadow-sm">
              ADMIN
            </span>
          </div>

          {/* User Info and Logout */}
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-xs text-white/80">Logged in as</p>
              <p className="font-medium text-white text-sm">{user.firstName} {user.lastName}</p>
            </div>
            <Button
              onClick={onLogout}
              variant="ghost"
              className="text-white hover:bg-white/20 border border-white/30 flex items-center gap-2 shadow-md p-2 h-auto"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Sign Out</span>
            </Button>
          </div>
        </div>
      </div>

      {/* Dashboard Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 max-w-4xl mx-auto">
        {/* User Management Card */}
        <Card
          className={cardClass}
          onClick={onNavigateToManageUsers}
        >
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="p-2.5 bg-white/20 rounded-lg shadow-inner">
                <Users className="w-5 h-5 text-white" />
              </div>
              <Button
                variant="ghost"
                className="text-white hover:bg-white/20 text-xs font-bold uppercase tracking-wider px-3 py-1 h-auto"
              >
                View
              </Button>
            </div>
            <h3 className="text-white font-semibold mb-1 text-sm uppercase tracking-wide opacity-90">User Management</h3>
            <p className="text-3xl font-bold text-white drop-shadow-sm">
              {loading ? '...' : stats.totalUsers.toLocaleString()} <span className="text-sm font-normal opacity-80">Users</span>
            </p>
          </CardContent>
        </Card>

        {/* Manage Doctors Card */}
        <Card
          className={cardClass}
          onClick={onNavigateToManageDoctors}
        >
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="p-2.5 bg-white/20 rounded-lg shadow-inner">
                <Stethoscope className="w-5 h-5 text-white" />
              </div>
              <Button
                variant="ghost"
                className="text-white hover:bg-white/20 text-xs font-bold uppercase tracking-wider px-3 py-1 h-auto"
              >
                View
              </Button>
            </div>
            <h3 className="text-white font-semibold mb-1 text-sm uppercase tracking-wide opacity-90">Manage Doctors</h3>
            <p className="text-3xl font-bold text-white drop-shadow-sm">
              {loading ? '...' : stats.totalDoctors.toLocaleString()} <span className="text-sm font-normal opacity-80">Doctors</span>
            </p>
          </CardContent>
        </Card>

        {/* Appointments Card */}
        <Card
          className={cardClass}
          onClick={onNavigateToAppointments}
        >
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="p-2.5 bg-white/20 rounded-lg shadow-inner">
                <Calendar className="w-5 h-5 text-white" />
              </div>
              <Button
                variant="ghost"
                className="text-white hover:bg-white/20 text-xs font-bold uppercase tracking-wider px-3 py-1 h-auto"
              >
                Manage
              </Button>
            </div>
            <h3 className="text-white font-semibold mb-1 text-sm uppercase tracking-wide opacity-90">Appointments</h3>
            <p className="text-3xl font-bold text-white drop-shadow-sm">
              {loading ? '...' : stats.pendingAppointments.toLocaleString()} <span className="text-sm font-normal opacity-80">Total</span>
            </p>
          </CardContent>
        </Card>

        {/* Analytics Card */}
        <Card
          className={cardClass}
          onClick={onNavigateToAnalytics}
        >
          <CardContent className="p-5">
            <div className="flex items-center justify-between mb-3">
              <div className="p-2.5 bg-white/20 rounded-lg shadow-inner">
                <TrendingUp className="w-5 h-5 text-white" />
              </div>
              <Button
                variant="ghost"
                className="text-white hover:bg-white/20 text-xs font-bold uppercase tracking-wider px-3 py-1 h-auto"
              >
                Details
              </Button>
            </div>
            <h3 className="text-white font-semibold mb-1 text-sm uppercase tracking-wide opacity-90">Analytics</h3>
            <p className="text-xl font-bold text-white drop-shadow-sm mt-1">
              {stats.monthlyReports}
            </p>
            <p className="text-xs text-white/70">Monthly Reports</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}