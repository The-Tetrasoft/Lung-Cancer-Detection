import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Calendar, Clock, User, Stethoscope, Search, Loader2, CheckCircle, XCircle, AlertCircle, ArrowLeft } from 'lucide-react';
import { BACKEND_BASE } from '../utils/config';
import { Alert, AlertDescription } from './ui/alert';

interface Appointment {
    id: string;
    username: string;
    patientName?: string;
    doctorId: string;
    doctorName: string;
    date: string;
    time: string;
    reason: string;
    status: 'pending' | 'confirmed' | 'cancelled' | 'completed';
    created_at: string;
}

interface ManageAppointmentsProps {
    onBack: () => void;
}

export function ManageAppointments({ onBack }: ManageAppointmentsProps) {
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [filteredAppointments, setFilteredAppointments] = useState<Appointment[]>([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [statusFilter, setStatusFilter] = useState<string>('all');
    const [updating, setUpdating] = useState<string | null>(null);

    useEffect(() => {
        fetchAppointments();
    }, []);

    useEffect(() => {
        filterAppointments();
    }, [appointments, searchTerm, statusFilter]);

    const fetchAppointments = async () => {
        try {
            setLoading(true);
            const response = await fetch(`${BACKEND_BASE}/api/appointments`);
            const data = await response.json();

            if (data.success) {
                setAppointments(data.appointments || []);
            }
        } catch (error) {
            console.error('Error fetching appointments:', error);
        } finally {
            setLoading(false);
        }
    };

    const filterAppointments = () => {
        let filtered = [...appointments];

        // Filter by search term
        if (searchTerm) {
            filtered = filtered.filter(apt =>
                apt.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
                apt.doctorName.toLowerCase().includes(searchTerm.toLowerCase())
            );
        }

        // Filter by status
        if (statusFilter !== 'all') {
            filtered = filtered.filter(apt => apt.status === statusFilter);
        }

        setFilteredAppointments(filtered);
    };

    const updateAppointmentStatus = async (id: string, newStatus: 'confirmed' | 'cancelled' | 'completed') => {
        try {
            setUpdating(id);
            const response = await fetch(`${BACKEND_BASE}/api/appointments/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: newStatus })
            });

            const data = await response.json();

            if (data.success) {
                // Update local state
                setAppointments(prev =>
                    prev.map(apt => apt.id === id ? { ...apt, status: newStatus } : apt)
                );
            }
        } catch (error) {
            console.error('Error updating appointment:', error);
        } finally {
            setUpdating(null);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'pending': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
            case 'confirmed': return 'bg-green-100 text-green-800 border-green-200';
            case 'cancelled': return 'bg-red-100 text-red-800 border-red-200';
            case 'completed': return 'bg-blue-100 text-blue-800 border-blue-200';
            default: return 'bg-gray-100 text-gray-800 border-gray-200';
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'pending': return <AlertCircle className="h-4 w-4" />;
            case 'confirmed': return <CheckCircle className="h-4 w-4" />;
            case 'cancelled': return <XCircle className="h-4 w-4" />;
            case 'completed': return <CheckCircle className="h-4 w-4" />;
            default: return null;
        }
    };

    const pendingCount = appointments.filter(a => a.status === 'pending').length;

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-8">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Button
                            onClick={onBack}
                            variant="outline"
                            className="border-blue-200 text-blue-700 hover:bg-blue-50"
                        >
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            Back
                        </Button>
                        <div>
                            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
                                Appointment Management
                            </h1>
                            <p className="text-slate-600">Review and manage patient appointments</p>
                        </div>
                    </div>
                    {pendingCount > 0 && (
                        <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200 px-4 py-2 text-base">
                            {pendingCount} Pending Request{pendingCount !== 1 ? 's' : ''}
                        </Badge>
                    )}
                </div>

                {/* Filters */}
                <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
                    <CardContent className="pt-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
                                <Input
                                    placeholder="Search by patient or doctor name..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    className="pl-10 border-blue-100 focus:border-blue-400"
                                />
                            </div>
                            <Select value={statusFilter} onValueChange={setStatusFilter}>
                                <SelectTrigger className="border-blue-100 focus:border-blue-400">
                                    <SelectValue placeholder="Filter by status" />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="all">All Statuses</SelectItem>
                                    <SelectItem value="pending">Pending</SelectItem>
                                    <SelectItem value="confirmed">Confirmed</SelectItem>
                                    <SelectItem value="completed">Completed</SelectItem>
                                    <SelectItem value="cancelled">Cancelled</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>
                    </CardContent>
                </Card>

                {/* Appointments List */}
                {loading ? (
                    <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
                        <CardContent className="p-12 flex items-center justify-center">
                            <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                        </CardContent>
                    </Card>
                ) : filteredAppointments.length === 0 ? (
                    <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
                        <CardContent className="p-12 text-center">
                            <Calendar className="h-16 w-16 text-slate-300 mx-auto mb-4" />
                            <h3 className="text-lg font-semibold text-slate-900 mb-2">No Appointments Found</h3>
                            <p className="text-slate-600">
                                {searchTerm || statusFilter !== 'all'
                                    ? 'Try adjusting your filters'
                                    : 'No appointments have been scheduled yet'}
                            </p>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="grid gap-4">
                        {filteredAppointments.map((appointment) => (
                            <Card key={appointment.id} className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md hover:shadow-lg transition-shadow">
                                <CardContent className="p-6">
                                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
                                        {/* Patient Info */}
                                        <div className="lg:col-span-3">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
                                                    <User className="h-5 w-5 text-blue-600" />
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-900">{appointment.patientName || appointment.username}</p>
                                                    <p className="text-xs text-slate-600">Patient</p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Doctor Info */}
                                        <div className="lg:col-span-3">
                                            <div className="flex items-center gap-3">
                                                <div className="w-10 h-10 rounded-full bg-teal-100 flex items-center justify-center">
                                                    <Stethoscope className="h-5 w-5 text-teal-600" />
                                                </div>
                                                <div>
                                                    <p className="font-semibold text-slate-900">{appointment.doctorName}</p>
                                                    <p className="text-xs text-slate-600">Doctor</p>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Date & Time */}
                                        <div className="lg:col-span-2">
                                            <div className="space-y-1">
                                                <div className="flex items-center gap-2 text-sm text-slate-700">
                                                    <Calendar className="h-4 w-4 text-blue-600" />
                                                    {new Date(appointment.date).toLocaleDateString()}
                                                </div>
                                                <div className="flex items-center gap-2 text-sm text-slate-700">
                                                    <Clock className="h-4 w-4 text-blue-600" />
                                                    {appointment.time}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Status */}
                                        <div className="lg:col-span-2">
                                            <Badge className={`${getStatusColor(appointment.status)} flex items-center gap-1 w-fit`}>
                                                {getStatusIcon(appointment.status)}
                                                {appointment.status.charAt(0).toUpperCase() + appointment.status.slice(1)}
                                            </Badge>
                                        </div>

                                        {/* Actions */}
                                        <div className="lg:col-span-2 flex gap-2">
                                            {appointment.status === 'pending' && (
                                                <>
                                                    <Button
                                                        size="sm"
                                                        onClick={() => updateAppointmentStatus(appointment.id, 'confirmed')}
                                                        disabled={updating === appointment.id}
                                                        className="bg-green-600 hover:bg-green-700 text-white flex-1"
                                                    >
                                                        {updating === appointment.id ? (
                                                            <Loader2 className="h-4 w-4 animate-spin" />
                                                        ) : (
                                                            'Confirm'
                                                        )}
                                                    </Button>
                                                    <Button
                                                        size="sm"
                                                        variant="outline"
                                                        onClick={() => updateAppointmentStatus(appointment.id, 'cancelled')}
                                                        disabled={updating === appointment.id}
                                                        className="border-red-200 text-red-700 hover:bg-red-50 flex-1"
                                                    >
                                                        Decline
                                                    </Button>
                                                </>
                                            )}
                                            {appointment.status === 'confirmed' && (
                                                <Button
                                                    size="sm"
                                                    onClick={() => updateAppointmentStatus(appointment.id, 'completed')}
                                                    disabled={updating === appointment.id}
                                                    className="bg-blue-600 hover:bg-blue-700 text-white w-full"
                                                >
                                                    {updating === appointment.id ? (
                                                        <Loader2 className="h-4 w-4 animate-spin" />
                                                    ) : (
                                                        'Mark Complete'
                                                    )}
                                                </Button>
                                            )}
                                        </div>
                                    </div>

                                    {/* Reason */}
                                    {appointment.reason && (
                                        <div className="mt-4 pt-4 border-t border-blue-100">
                                            <p className="text-sm text-slate-600">
                                                <span className="font-semibold">Reason:</span> {appointment.reason}
                                            </p>
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
