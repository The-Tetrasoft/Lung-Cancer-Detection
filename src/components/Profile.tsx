import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { User } from 'lucide-react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import { BACKEND_BASE } from '../utils/config';

interface ProfileProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
    profilePicture?: string;
    phoneNumber?: string;
    dateOfBirth?: string;
    gender?: 'male' | 'female' | 'other';
  };
  onUserUpdate?: (user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
    profilePicture?: string | null;
    phoneNumber?: string | null;
    dateOfBirth?: string | null;
    gender?: 'male' | 'female' | 'other' | null;
  }) => void;
  onNavigateBack: () => void;
}

export function Profile({ user, onUserUpdate, onNavigateBack }: ProfileProps) {
  const initials = `${user.firstName?.[0] || ''}${user.lastName?.[0] || ''}`.toUpperCase();
  const [editing, setEditing] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState('');
  const [success, setSuccess] = React.useState('');
  const [form, setForm] = React.useState({
    firstName: user.firstName || '',
    lastName: user.lastName || '',
    email: user.email || '',
    phoneNumber: user.phoneNumber || '',
    dateOfBirth: user.dateOfBirth || '',
    gender: (user.gender as string) || ''
  });

  const handleChange = (name: string, value: string) => {
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const res = await fetch(`${BACKEND_BASE}/api/profile/${encodeURIComponent(user.username)}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          firstName: form.firstName,
          lastName: form.lastName,
          email: form.email,
          phoneNumber: form.phoneNumber,
          dateOfBirth: form.dateOfBirth,
          gender: form.gender
        })
      });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data?.message || 'Failed to update profile');
      }
      setSuccess('Profile updated successfully');
      setEditing(false);
      onUserUpdate?.(data.user);
    } catch (e: any) {
      setError(e.message || 'Update failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <div className="flex justify-start mb-4">
            <Button variant="ghost" onClick={onNavigateBack} className="text-slate-600 hover:text-blue-600">
              ← Back to Dashboard
            </Button>
          </div>
          <div className="mx-auto w-16 h-16 mb-4 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center shadow-inner">
            <User className="w-8 h-8 text-blue-600" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
            My Profile
          </h1>
          <p className="text-slate-600 mt-2">Manage your account settings and personal information</p>
        </div>

        {error && (
          <Alert className="border-red-200 bg-red-50 text-red-800">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {success && (
          <Alert className="border-teal-200 bg-teal-50 text-teal-800">
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-lg overflow-hidden">
          <CardHeader className="bg-gradient-to-r from-blue-50/50 to-indigo-50/50 border-b border-blue-100">
            <CardTitle className="flex items-center text-slate-800">
              <User className="w-5 h-5 mr-2 text-blue-600" />
              Account Information
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="flex items-center gap-6 mb-8 pb-8 border-b border-blue-50">
              <Avatar className="w-20 h-20 border-4 border-white shadow-md ring-2 ring-blue-100">
                <AvatarImage src={user.profilePicture} />
                <AvatarFallback className="bg-gradient-to-br from-blue-500 to-indigo-600 text-white text-xl">
                  {initials || 'U'}
                </AvatarFallback>
              </Avatar>
              <div>
                <h3 className="text-2xl font-bold text-slate-900">
                  {user.firstName} {user.lastName}
                </h3>
                <p className="text-blue-600 font-medium">@{user.username}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 capitalize border border-blue-200">
                    {user.role}
                  </span>
                </div>
              </div>
            </div>

            {!editing ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Username</p>
                  <p className="font-semibold text-slate-900">{user.username}</p>
                </div>
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Email</p>
                  <p className="font-semibold text-slate-900">{user.email}</p>
                </div>
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Role</p>
                  <p className="font-semibold text-slate-900 capitalize">{user.role}</p>
                </div>
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Phone</p>
                  <p className="font-semibold text-slate-900">{user.phoneNumber || '—'}</p>
                </div>
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Date of Birth</p>
                  <p className="font-semibold text-slate-900">{user.dateOfBirth ? new Date(user.dateOfBirth).toLocaleDateString() : '—'}</p>
                </div>
                <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100">
                  <p className="text-sm text-slate-500 mb-1">Gender</p>
                  <p className="font-semibold text-slate-900 capitalize">{user.gender || '—'}</p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-slate-700">First Name</Label>
                    <Input
                      value={form.firstName}
                      onChange={(e) => handleChange('firstName', e.target.value)}
                      className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700">Last Name</Label>
                    <Input
                      value={form.lastName}
                      onChange={(e) => handleChange('lastName', e.target.value)}
                      className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700">Email</Label>
                    <Input
                      type="email"
                      value={form.email}
                      onChange={(e) => handleChange('email', e.target.value)}
                      className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700">Phone</Label>
                    <Input
                      value={form.phoneNumber}
                      onChange={(e) => handleChange('phoneNumber', e.target.value)}
                      className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700">Date of Birth</Label>
                    <Input
                      type="date"
                      value={form.dateOfBirth}
                      onChange={(e) => handleChange('dateOfBirth', e.target.value)}
                      className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-slate-700">Gender</Label>
                    <Select value={form.gender} onValueChange={(v) => handleChange('gender', v)}>
                      <SelectTrigger className="border-blue-100 focus:border-blue-400 focus:ring-blue-400/20">
                        <SelectValue placeholder="Select gender" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            )}

            <div className="mt-8 flex gap-3 justify-end border-t border-blue-50 pt-6">
              {!editing ? (
                <Button
                  onClick={() => setEditing(true)}
                  className="bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg transition-all"
                >
                  Edit Profile
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={() => {
                      setEditing(false); setError(''); setSuccess(''); setForm({
                        firstName: user.firstName || '',
                        lastName: user.lastName || '',
                        email: user.email || '',
                        phoneNumber: user.phoneNumber || '',
                        dateOfBirth: user.dateOfBirth || '',
                        gender: (user.gender as string) || ''
                      });
                    }}
                    className="border-slate-200 hover:bg-slate-50 text-slate-700"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSave}
                    disabled={saving}
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md"
                  >
                    {saving ? 'Saving...' : 'Save Changes'}
                  </Button>
                </>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
