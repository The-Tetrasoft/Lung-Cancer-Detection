import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from './ui/table';
import {
    AlertDialog,
    AlertDialogAction,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from './ui/alert-dialog';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from './ui/select';
import { Search, UserPlus, Edit2, Trash2, Shield, User, Mail, Phone, Calendar, ArrowLeft, Eye } from 'lucide-react';
import { db } from '../firebase-client';
import { collection, getDocs, deleteDoc, doc, updateDoc, query, where } from 'firebase/firestore';

interface User {
    id?: string;
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    phoneNumber: string;
    dateOfBirth: string;
    gender: string;
    role: 'user' | 'admin';
    profilePicture?: string;
    created_at: string;
    loginAttempts?: number;
}

interface ManageUsersProps {
    onNavigateBack: () => void;
}

export function ManageUsers({ onNavigateBack }: ManageUsersProps) {
    const [users, setUsers] = useState<User[]>([]);
    const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [roleFilter, setRoleFilter] = useState<'all' | 'user' | 'admin'>('all');
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [userToDelete, setUserToDelete] = useState<User | null>(null);
    const [viewingUser, setViewingUser] = useState<User | null>(null);

    // Fetch users from API
    useEffect(() => {
        fetchUsers();
    }, []);

    const fetchUsers = async () => {
        setIsLoading(true);
        setError('');
        try {
            const usersRef = collection(db, 'users');
            const snapshot = await getDocs(usersRef);

            const fetchedUsers: User[] = snapshot.docs.map(doc => {
                const data = doc.data();
                return {
                    id: doc.id,
                    username: data.username || '',
                    firstName: data.firstName || '',
                    lastName: data.lastName || '',
                    email: data.email || '',
                    phoneNumber: data.phoneNumber || '',
                    dateOfBirth: data.dateOfBirth || '',
                    gender: data.gender || '',
                    role: data.role || 'user',
                    profilePicture: data.profilePicture || '',
                    created_at: data.createdAt || data.created_at || new Date().toISOString(),
                    loginAttempts: data.loginAttempts || 0
                } as User;
            });

            setUsers(fetchedUsers);
            setFilteredUsers(fetchedUsers);
        } catch (err) {
            console.error('Error fetching users:', err);
            setError('Failed to load users from database.');
        } finally {
            setIsLoading(false);
        }
    };

    // Filter users based on search and role
    useEffect(() => {
        let filtered = users;

        // Filter by search query
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(user =>
                user.username.toLowerCase().includes(query) ||
                user.firstName.toLowerCase().includes(query) ||
                user.lastName.toLowerCase().includes(query) ||
                user.email.toLowerCase().includes(query)
            );
        }

        // Filter by role
        if (roleFilter !== 'all') {
            filtered = filtered.filter(user => user.role === roleFilter);
        }

        setFilteredUsers(filtered);
    }, [searchQuery, roleFilter, users]);

    const handleDeleteUser = async (user: User) => {
        try {
            if (user.id) {
                await deleteDoc(doc(db, 'users', user.id));
            } else {
                // Fallback if no ID (shouldn't happen with new fetch)
                // Query by username to find doc
                const q = query(collection(db, 'users'), where('username', '==', user.username));
                const snap = await getDocs(q);
                if (!snap.empty) {
                    await deleteDoc(doc(db, 'users', snap.docs[0].id));
                }
            }

            // Remove from local state
            setUsers(users.filter(u => u.username !== user.username));
            setDeleteDialogOpen(false);
            setUserToDelete(null);
        } catch (err) {
            console.error('Error deleting user:', err);
            setError('Failed to delete user. Please try again.');
        }
    };

    const handleRoleChange = async (user: User, newRole: 'user' | 'admin') => {
        try {
            if (user.id) {
                await updateDoc(doc(db, 'users', user.id), { role: newRole });
            } else {
                const q = query(collection(db, 'users'), where('username', '==', user.username));
                const snap = await getDocs(q);
                if (!snap.empty) {
                    await updateDoc(doc(db, 'users', snap.docs[0].id), { role: newRole });
                }
            }

            // Update local state
            setUsers(users.map(u =>
                u.username === user.username ? { ...u, role: newRole } : u
            ));
        } catch (err) {
            console.error('Error updating role:', err);
            setError('Failed to update user role. Please try again.');
        }
    };

    const formatDate = (dateString: string) => {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-6">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <Button
                            variant="ghost"
                            onClick={onNavigateBack}
                            className="mb-4 text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                        >
                            <ArrowLeft className="w-4 h-4 mr-2" />
                            Back to Dashboard
                        </Button>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">User Management</h1>
                        <p className="text-slate-600 mt-1">Manage user accounts, permissions, and access controls</p>
                    </div>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-slate-600">Total Users</p>
                                    <p className="text-2xl font-bold text-blue-600">{users.length}</p>
                                </div>
                                <User className="w-8 h-8 text-blue-600" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-white/80 backdrop-blur-sm border-indigo-100 shadow-sm hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-slate-600">Administrators</p>
                                    <p className="text-2xl font-bold text-indigo-600">{users.filter(u => u.role === 'admin').length}</p>
                                </div>
                                <Shield className="w-8 h-8 text-indigo-600" />
                            </div>
                        </CardContent>
                    </Card>
                    <Card className="bg-white/80 backdrop-blur-sm border-teal-100 shadow-sm hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-slate-600">Regular Users</p>
                                    <p className="text-2xl font-bold text-teal-600">{users.filter(u => u.role === 'user').length}</p>
                                </div>
                                <UserPlus className="w-8 h-8 text-teal-600" />
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Main Content */}
                <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-lg">
                    <CardHeader>
                        <CardTitle className="text-slate-900">Users Directory</CardTitle>
                        <CardDescription className="text-slate-500">
                            View and manage all registered users
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {/* Filters */}
                        <div className="flex flex-col md:flex-row gap-4 mb-6">
                            <div className="flex-1 relative">
                                <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                                <Input
                                    placeholder="Search by username, name, or email..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="pl-10 border-blue-100 focus:border-blue-400"
                                />
                            </div>
                            <Select value={roleFilter} onValueChange={(value: any) => setRoleFilter(value)}>
                                <SelectTrigger className="w-full md:w-48 border-blue-100">
                                    <SelectValue placeholder="Filter by role" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Roles</SelectItem>
                                    <SelectItem value="user">Users</SelectItem>
                                    <SelectItem value="admin">Admins</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {error && (
                            <Alert className="mb-4 border-red-200 bg-red-50">
                                <AlertDescription className="text-red-800">{error}</AlertDescription>
                            </Alert>
                        )}

                        {isLoading ? (
                            <div className="text-center py-12">
                                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                                <p className="mt-4 text-slate-600">Loading users...</p>
                            </div>
                        ) : filteredUsers.length === 0 ? (
                            <div className="text-center py-12">
                                <User className="w-12 h-12 text-slate-400 mx-auto mb-4" />
                                <p className="text-slate-600">No users found</p>
                            </div>
                        ) : (
                            <div className="overflow-x-auto">
                                <Table>
                                    <TableHeader>
                                        <TableRow className="hover:bg-blue-50/50 border-blue-100">
                                            <TableHead className="text-slate-700">User</TableHead>
                                            <TableHead className="text-slate-700">Email</TableHead>
                                            <TableHead className="text-slate-700">Phone</TableHead>
                                            <TableHead className="text-slate-700">Role</TableHead>
                                            <TableHead className="text-slate-700">Joined</TableHead>
                                            <TableHead className="text-right text-slate-700">Actions</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {filteredUsers.map((user) => (
                                            <TableRow key={user.username} className="hover:bg-blue-50/30 border-blue-50">
                                                <TableCell>
                                                    <div className="flex items-center gap-3">
                                                        <Avatar className="border-2 border-blue-100">
                                                            <AvatarImage src={user.profilePicture} />
                                                            <AvatarFallback className="bg-blue-100 text-blue-700">
                                                                {user.firstName[0]}{user.lastName[0]}
                                                            </AvatarFallback>
                                                        </Avatar>
                                                        <div>
                                                            <p className="font-medium text-slate-900">{user.firstName} {user.lastName}</p>
                                                            <p className="text-sm text-slate-500">@{user.username}</p>
                                                        </div>
                                                    </div>
                                                </TableCell>
                                                <TableCell>
                                                    <div className="flex items-center gap-2">
                                                        <Mail className="w-4 h-4 text-slate-400" />
                                                        <span className="text-sm text-slate-600">{user.email}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell>
                                                    <div className="flex items-center gap-2">
                                                        <Phone className="w-4 h-4 text-slate-400" />
                                                        <span className="text-sm text-slate-600">{user.phoneNumber || 'N/A'}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell>
                                                    <Select
                                                        value={user.role}
                                                        onValueChange={(value: 'user' | 'admin') => handleRoleChange(user, value)}
                                                    >
                                                        <SelectTrigger className="w-32 border-blue-100 h-8">
                                                            <SelectValue />
                                                        </SelectTrigger>
                                                        <SelectContent>
                                                            <SelectItem value="user">
                                                                <div className="flex items-center gap-2 text-slate-700">
                                                                    <User className="w-4 h-4 text-blue-500" />
                                                                    User
                                                                </div>
                                                            </SelectItem>
                                                            <SelectItem value="admin">
                                                                <div className="flex items-center gap-2 text-slate-700">
                                                                    <Shield className="w-4 h-4 text-indigo-500" />
                                                                    Admin
                                                                </div>
                                                            </SelectItem>
                                                        </SelectContent>
                                                    </Select>
                                                </TableCell>
                                                <TableCell>
                                                    <div className="flex items-center gap-2">
                                                        <Calendar className="w-4 h-4 text-slate-400" />
                                                        <span className="text-sm text-slate-600">{formatDate(user.created_at)}</span>
                                                    </div>
                                                </TableCell>
                                                <TableCell className="text-right">
                                                    <div className="flex items-center justify-end gap-2">
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => setViewingUser(user)}
                                                            className="text-blue-600 hover:text-blue-700 hover:bg-blue-50"
                                                        >
                                                            <Eye className="w-4 h-4" />
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => {
                                                                setUserToDelete(user);
                                                                setDeleteDialogOpen(true);
                                                            }}
                                                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </Button>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>
                        )}
                    </CardContent>
                </Card>
            </div>

            {/* Delete Confirmation Dialog */}
            <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
                <AlertDialogContent className="bg-white/95 backdrop-blur-sm border-red-100">
                    <AlertDialogHeader>
                        <AlertDialogTitle className="text-red-600">Delete User</AlertDialogTitle>
                        <AlertDialogDescription className="text-slate-600">
                            Are you sure you want to delete this user? This action cannot be undone.
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel className="border-slate-200 hover:bg-slate-50">Cancel</AlertDialogCancel>
                        <AlertDialogAction
                            onClick={() => userToDelete && handleDeleteUser(userToDelete)}
                            className="bg-red-600 hover:bg-red-700 border-0"
                        >
                            Delete
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>

            {/* User Details Dialog */}
            {viewingUser && (
                <AlertDialog open={!!viewingUser} onOpenChange={() => setViewingUser(null)}>
                    <AlertDialogContent className="max-w-2xl bg-white/95 backdrop-blur-sm border-blue-100">
                        <AlertDialogHeader>
                            <AlertDialogTitle className="text-slate-900">User Details</AlertDialogTitle>
                        </AlertDialogHeader>
                        <div className="space-y-4">
                            <div className="flex items-center gap-4">
                                <Avatar className="w-20 h-20 border-4 border-blue-50 ring-2 ring-blue-100">
                                    <AvatarImage src={viewingUser.profilePicture} />
                                    <AvatarFallback className="text-2xl bg-blue-100 text-blue-700">
                                        {viewingUser.firstName[0]}{viewingUser.lastName[0]}
                                    </AvatarFallback>
                                </Avatar>
                                <div>
                                    <h3 className="text-xl font-bold text-slate-900">{viewingUser.firstName} {viewingUser.lastName}</h3>
                                    <p className="text-slate-500">@{viewingUser.username}</p>
                                    <Badge className={`mt-2 ${viewingUser.role === 'admin' ? 'bg-indigo-100 text-indigo-700 hover:bg-indigo-200' : 'bg-blue-100 text-blue-700 hover:bg-blue-200'}`}>
                                        {viewingUser.role === 'admin' ? 'Administrator' : 'User'}
                                    </Badge>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-lg border border-slate-100">
                                <div>
                                    <p className="text-sm text-slate-500">Email</p>
                                    <p className="font-medium text-slate-900">{viewingUser.email}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-500">Phone</p>
                                    <p className="font-medium text-slate-900">{viewingUser.phoneNumber || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-500">Date of Birth</p>
                                    <p className="font-medium text-slate-900">{formatDate(viewingUser.dateOfBirth)}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-500">Gender</p>
                                    <p className="font-medium capitalize text-slate-900">{viewingUser.gender || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-500">Account Created</p>
                                    <p className="font-medium text-slate-900">{formatDate(viewingUser.created_at)}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-slate-500">Login Attempts</p>
                                    <p className="font-medium text-slate-900">{viewingUser.loginAttempts || 0}</p>
                                </div>
                            </div>
                        </div>
                        <AlertDialogFooter>
                            <AlertDialogCancel className="border-blue-200 text-blue-700 hover:bg-blue-50">Close</AlertDialogCancel>
                        </AlertDialogFooter>
                    </AlertDialogContent>
                </AlertDialog>
            )}
        </div>
    );
}
