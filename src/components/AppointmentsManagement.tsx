import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ArrowLeft, Calendar, Clock, User, Stethoscope, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { BACKEND_BASE } from '../utils/config';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from './ui/dialog';
import { Alert, AlertDescription } from './ui/alert';

interface Appointment {
    id: string;
    username: string;
    patientName: string;
    doctorId: string;
    doctorName: string;
    date: string;
    time: string;
    reason: string;
    status: 'pending' | 'confirmed' | 'cancelled' | 'completed';
    createdAt: string;
}

interface AppointmentsManagementProps {
    onNavigateBack: () => void;
}

export function AppointmentsManagement({ onNavigateBack }: AppointmentsManagementProps) {
    const [appointments, setAppointments] = useState<Appointment[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);
    const [showDetailsDialog, setShowDetailsDialog] = useState(false);
    const [updating, setUpdating] = useState(false);
    const [filter, setFilter] = useState<'all' | 'pending' | 'confirmed' | 'cancelled' | 'completed'>('all');

    useEffect(() => {
        fetchAppointments();
    }, []);

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

    const updateAppointmentStatus = async (appointmentId: string, newStatus: 'confirmed' | 'cancelled' | 'completed') => {
        try {
            setUpdating(true);
            const response = await fetch(`${BACKEND_BASE}/api/appointments/${appointmentId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: newStatus })
            });

            const data = await response.json();

            if (data.success) {
                // Update local state
                setAppointments(prev =>
                    prev.map(appt =>
                        appt.id === appointmentId ? { ...appt, status: newStatus } : appt
                    )
                );
                setShowDetailsDialog(false);
            }
        } catch (error) {
            console.error('Error updating appointment:', error);
        } finally {
            setUpdating(false);
        }
    };

    const handleViewDetails = (appointment: Appointment) => {
        setSelectedAppointment(appointment);
        setShowDetailsDialog(true);
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
            default: return <AlertCircle className="h-4 w-4" />;
        }
    };

    const filteredAppointments = filter === 'all'
        ? appointments
        : appointments.filter(appt => appt.status === filter);

    const stats = {
        total: appointments.length,
        pending: appointments.filter(a => a.status === 'pending').length,
        confirmed: appointments.filter(a => a.status === 'confirmed').length,
        completed: appointments.filter(a => a.status === 'completed').length,
        cancelled: appointments.filter(a => a.status === 'cancelled').length,
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 md:p-8 flex items-center justify-center">
                <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 md:p-8">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-6">
                    <Button
                        onClick={onNavigateBack}
                        variant="ghost"
                        className="mb-4 hover:bg-white/50"
                    >
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Admin Panel
                    </Button>

                    <div className="bg-white/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-blue-100">
                        <div className="flex items-center gap-3">
                            <div className="p-3 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg shadow-md">
                                <Calendar className="h-6 w-6 text-white" />
                            </div>
                            <div>
                                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
                                    Appointments Management
                                </h1>
                                <p className="text-slate-600">View and manage patient appointments</p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Statistics */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    <Card className="bg-white/80 backdrop-blur-sm shadow-sm border-slate-200">
                        <CardContent className="pt-6">
                            <div className="text-center">
                                <div className="text-2xl font-bold text-slate-900">{stats.total}</div>
                                <p className="text-sm text-slate-600">Total</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-white/80 backdrop-blur-sm shadow-sm border-yellow-200">
                        <CardContent className="pt-6">
                            <div className="text-center">
                                <div className="text-2xl font-bold text-yellow-600">{stats.pending}</div>
                                <p className="text-sm text-slate-600">Pending</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-white/80 backdrop-blur-sm shadow-sm border-green-200">
                        <CardContent className="pt-6">
                            <div className="text-center">
                                <div className="text-2xl font-bold text-green-600">{stats.confirmed}</div>
                                <p className="text-sm text-slate-600">Confirmed</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-white/80 backdrop-blur-sm shadow-sm border-blue-200">
                        <CardContent className="pt-6">
                            <div className="text-center">
                                <div className="text-2xl font-bold text-blue-600">{stats.completed}</div>
                                <p className="text-sm text-slate-600">Completed</p>
                            </div>
                        </CardContent>
                    </Card>

                    <Card className="bg-white/80 backdrop-blur-sm shadow-sm border-red-200">
                        <CardContent className="pt-6">
                            <div className="text-center">
                                <div className="text-2xl font-bold text-red-600">{stats.cancelled}</div>
                                <p className="text-sm text-slate-600">Cancelled</p>
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* Filter Buttons */}
                <div className="mb-6 flex flex-wrap gap-2">
                    <Button
                        onClick={() => setFilter('all')}
                        variant={filter === 'all' ? 'default' : 'outline'}
                        className={filter === 'all' ? 'bg-blue-600' : ''}
                    >
                        All Appointments
                    </Button>
                    <Button
                        onClick={() => setFilter('pending')}
                        variant={filter === 'pending' ? 'default' : 'outline'}
                        className={filter === 'pending' ? 'bg-yellow-600 hover:bg-yellow-700' : ''}
                    >
                        Pending
                    </Button>
                    <Button
                        onClick={() => setFilter('confirmed')}
                        variant={filter === 'confirmed' ? 'default' : 'outline'}
                        className={filter === 'confirmed' ? 'bg-green-600 hover:bg-green-700' : ''}
                    >
                        Confirmed
                    </Button>
                    <Button
                        onClick={() => setFilter('completed')}
                        variant={filter === 'completed' ? 'default' : 'outline'}
                        className={filter === 'completed' ? 'bg-blue-600 hover:bg-blue-700' : ''}
                    >
                        Completed
                    </Button>
                    <Button
                        onClick={() => setFilter('cancelled')}
                        variant={filter === 'cancelled' ? 'default' : 'outline'}
                        className={filter === 'cancelled' ? 'bg-red-600 hover:bg-red-700' : ''}
                    >
                        Cancelled
                    </Button>
                </div>

                {/* Appointments List */}
                <Card className="bg-white/80 backdrop-blur-sm shadow-lg">
                    <CardHeader>
                        <CardTitle>Appointments ({filteredAppointments.length})</CardTitle>
                        <CardDescription>
                            {filter === 'all' ? 'All appointments' : `${filter.charAt(0).toUpperCase() + filter.slice(1)} appointments`}
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        {filteredAppointments.length === 0 ? (
                            <div className="text-center py-12">
                                <Calendar className="h-16 w-16 mx-auto text-gray-300 mb-4" />
                                <p className="text-gray-500">No appointments found</p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {filteredAppointments.map((appointment) => (
                                    <Card key={appointment.id} className="border-blue-100 hover:shadow-md transition-shadow">
                                        <CardContent className="p-4">
                                            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                                                <div className="flex-1 space-y-2">
                                                    <div className="flex items-center gap-2 flex-wrap">
                                                        <Badge className={getStatusColor(appointment.status)}>
                                                            {getStatusIcon(appointment.status)}
                                                            <span className="ml-1">{appointment.status.toUpperCase()}</span>
                                                        </Badge>
                                                        <span className="text-sm text-gray-500">
                                                            {new Date(appointment.createdAt).toLocaleDateString()}
                                                        </span>
                                                    </div>

                                                    <div className="grid md:grid-cols-2 gap-3">
                                                        <div className="flex items-center gap-2">
                                                            <User className="h-4 w-4 text-blue-600" />
                                                            <div>
                                                                <p className="text-sm font-semibold text-slate-900">
                                                                    {appointment.patientName || appointment.username}
                                                                </p>
                                                                <p className="text-xs text-slate-500">Patient</p>
                                                            </div>
                                                        </div>

                                                        <div className="flex items-center gap-2">
                                                            <Stethoscope className="h-4 w-4 text-teal-600" />
                                                            <div>
                                                                <p className="text-sm font-semibold text-slate-900">
                                                                    {appointment.doctorName}
                                                                </p>
                                                                <p className="text-xs text-slate-500">Doctor</p>
                                                            </div>
                                                        </div>

                                                        <div className="flex items-center gap-2">
                                                            <Calendar className="h-4 w-4 text-purple-600" />
                                                            <div>
                                                                <p className="text-sm font-semibold text-slate-900">
                                                                    {new Date(appointment.date).toLocaleDateString()}
                                                                </p>
                                                                <p className="text-xs text-slate-500">Date</p>
                                                            </div>
                                                        </div>

                                                        <div className="flex items-center gap-2">
                                                            <Clock className="h-4 w-4 text-orange-600" />
                                                            <div>
                                                                <p className="text-sm font-semibold text-slate-900">
                                                                    {new Date(`2000-01-01T${appointment.time}`).toLocaleTimeString('en-US', {
                                                                        hour: 'numeric',
                                                                        minute: '2-digit',
                                                                        hour12: true
                                                                    })}
                                                                </p>
                                                                <p className="text-xs text-slate-500">Time</p>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {appointment.reason && (
                                                        <p className="text-sm text-slate-600 mt-2">
                                                            <span className="font-semibold">Reason:</span> {appointment.reason}
                                                        </p>
                                                    )}
                                                </div>

                                                <div className="flex md:flex-col gap-2">
                                                    <Button
                                                        onClick={() => handleViewDetails(appointment)}
                                                        variant="outline"
                                                        size="sm"
                                                        className="flex-1 md:flex-none"
                                                    >
                                                        View Details
                                                    </Button>
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                        )}
                    </CardContent>
                </Card>

                {/* Details Dialog */}
                <Dialog open={showDetailsDialog} onOpenChange={setShowDetailsDialog}>
                    <DialogContent className="sm:max-w-md">
                        <DialogHeader>
                            <DialogTitle>Appointment Details</DialogTitle>
                            <DialogDescription>
                                Manage appointment status and view details
                            </DialogDescription>
                        </DialogHeader>

                        {selectedAppointment && (
                            <div className="space-y-4">
                                <div className="p-4 bg-gray-50 rounded-lg space-y-3">
                                    <div>
                                        <p className="text-sm text-gray-600">Patient</p>
                                        <p className="font-semibold">{selectedAppointment.patientName || selectedAppointment.username}</p>
                                    </div>

                                    <div>
                                        <p className="text-sm text-gray-600">Doctor</p>
                                        <p className="font-semibold">{selectedAppointment.doctorName}</p>
                                    </div>

                                    <div className="grid grid-cols-2 gap-3">
                                        <div>
                                            <p className="text-sm text-gray-600">Date</p>
                                            <p className="font-semibold">{new Date(selectedAppointment.date).toLocaleDateString()}</p>
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-600">Time</p>
                                            <p className="font-semibold">
                                                {new Date(`2000-01-01T${selectedAppointment.time}`).toLocaleTimeString('en-US', {
                                                    hour: 'numeric',
                                                    minute: '2-digit',
                                                    hour12: true
                                                })}
                                            </p>
                                        </div>
                                    </div>

                                    <div>
                                        <p className="text-sm text-gray-600">Reason</p>
                                        <p className="font-semibold">{selectedAppointment.reason}</p>
                                    </div>

                                    <div>
                                        <p className="text-sm text-gray-600">Status</p>
                                        <Badge className={getStatusColor(selectedAppointment.status)}>
                                            {selectedAppointment.status.toUpperCase()}
                                        </Badge>
                                    </div>
                                </div>

                                {selectedAppointment.status === 'pending' && (
                                    <Alert>
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertDescription>
                                            This appointment is pending approval. Confirm or cancel it below.
                                        </AlertDescription>
                                    </Alert>
                                )}

                                <DialogFooter className="flex-col sm:flex-row gap-2">
                                    {selectedAppointment.status === 'pending' && (
                                        <>
                                            <Button
                                                onClick={() => updateAppointmentStatus(selectedAppointment.id, 'confirmed')}
                                                disabled={updating}
                                                className="bg-green-600 hover:bg-green-700 flex-1"
                                            >
                                                {updating ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <CheckCircle className="h-4 w-4 mr-2" />}
                                                Confirm
                                            </Button>
                                            <Button
                                                onClick={() => updateAppointmentStatus(selectedAppointment.id, 'cancelled')}
                                                disabled={updating}
                                                variant="destructive"
                                                className="flex-1"
                                            >
                                                {updating ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <XCircle className="h-4 w-4 mr-2" />}
                                                Cancel
                                            </Button>
                                        </>
                                    )}

                                    {selectedAppointment.status === 'confirmed' && (
                                        <Button
                                            onClick={() => updateAppointmentStatus(selectedAppointment.id, 'completed')}
                                            disabled={updating}
                                            className="bg-blue-600 hover:bg-blue-700 flex-1"
                                        >
                                            {updating ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <CheckCircle className="h-4 w-4 mr-2" />}
                                            Mark as Completed
                                        </Button>
                                    )}

                                    <Button
                                        onClick={() => setShowDetailsDialog(false)}
                                        variant="outline"
                                        className="flex-1"
                                    >
                                        Close
                                    </Button>
                                </DialogFooter>
                            </div>
                        )}
                    </DialogContent>
                </Dialog>
            </div>
        </div>
    );
}
