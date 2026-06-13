import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { UserCircle, MapPin, Stethoscope, Calendar, Clock, Loader2, CheckCircle, User, AlertCircle } from 'lucide-react';
import { BACKEND_BASE } from '../utils/config';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { db } from '../firebase-client';
import { collection, getDocs } from 'firebase/firestore';

interface Doctor {
    id: string;
    firstName: string;
    lastName: string;
    specialties: string[];
    hospital: string;
    location: string;
    experience: number;
    consultationFee: number;
    imageUrl?: string;
}

interface User {
    username: string;
    firstName: string;
    lastName: string;
}

interface AnalysisResult {
    prediction: 'Positive' | 'Negative';
    riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
}

interface DoctorRecommendationsProps {
    user: User;
    analysisResult: AnalysisResult;
    onNavigateToDoctor: (doctorId: string) => void;
}

export function DoctorRecommendations({ user, analysisResult, onNavigateToDoctor }: DoctorRecommendationsProps) {
    const [doctors, setDoctors] = useState<Doctor[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedDoctor, setSelectedDoctor] = useState<Doctor | null>(null);
    const [showBookingDialog, setShowBookingDialog] = useState(false);
    const [showProfileDialog, setShowProfileDialog] = useState(false);
    const [bookingData, setBookingData] = useState({
        date: '',
        time: '',
        reason: ''
    });
    const [isBooking, setIsBooking] = useState(false);
    const [bookingSuccess, setBookingSuccess] = useState(false);
    const [timeError, setTimeError] = useState<string>('');

    useEffect(() => {
        fetchRecommendedDoctors();
    }, []);

    const fetchRecommendedDoctors = async () => {
        try {
            setLoading(true);

            // Fetch from Firestore instead of API
            const doctorsRef = collection(db, 'doctors');
            const querySnapshot = await getDocs(doctorsRef);

            const fetchedDoctors: Doctor[] = querySnapshot.docs.map(doc => {
                const data = doc.data();
                return {
                    id: doc.id,
                    firstName: data.firstName || '',
                    lastName: data.lastName || '',
                    specialties: data.specialties || [],
                    hospital: data.hospital || '',
                    location: data.location || '',
                    experience: data.experience || 0,
                    consultationFee: data.consultationFee || 0,
                    imageUrl: data.imageUrl || ''
                } as Doctor;
            });

            if (fetchedDoctors.length > 0) {
                // Filter doctors by relevant specialties
                const relevantSpecialties = ['Oncology', 'Pulmonology', 'Thoracic Surgery', 'Radiology'];
                const filtered = fetchedDoctors.filter((doc: Doctor) =>
                    doc.specialties?.some(s => relevantSpecialties.some(rs => s.toLowerCase().includes(rs.toLowerCase())))
                );

                // If no relevant specialists found, show all doctors (fallback)
                const doctorsToShow = filtered.length > 0 ? filtered : fetchedDoctors;

                // Sort by experience and limit to top 3
                const sorted = doctorsToShow.sort((a: Doctor, b: Doctor) => b.experience - a.experience).slice(0, 3);
                setDoctors(sorted);
            }
        } catch (error) {
            console.error('Error fetching doctors:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleBookAppointment = (doctor: Doctor) => {
        console.log('Book appointment clicked for:', doctor.firstName, doctor.lastName);
        setSelectedDoctor(doctor);
        setBookingData({
            date: '',
            time: '',
            reason: `Follow-up consultation for lung cancer screening (${analysisResult.riskLevel} risk)`
        });
        setShowBookingDialog(true);
        setBookingSuccess(false);
        setTimeError('');
    };

    const handleViewProfile = (doctor: Doctor) => {
        console.log('View profile clicked for:', doctor.firstName, doctor.lastName);
        onNavigateToDoctor(doctor.id);
    };

    // Generate available time slots (10 AM - 4 PM)
    const generateTimeSlots = () => {
        const slots = [];
        for (let hour = 10; hour <= 16; hour++) {
            slots.push(`${hour.toString().padStart(2, '0')}:00`);
            if (hour < 16) {
                slots.push(`${hour.toString().padStart(2, '0')}:30`);
            }
        }
        return slots;
    };

    const validateTime = (time: string): boolean => {
        if (!time) return false;
        const [hours, minutes] = time.split(':').map(Number);
        const timeInMinutes = hours * 60 + minutes;
        const startTime = 10 * 60; // 10:00 AM
        const endTime = 16 * 60; // 4:00 PM

        if (timeInMinutes < startTime || timeInMinutes > endTime) {
            setTimeError('Please select a time between 10:00 AM and 4:00 PM');
            return false;
        }
        setTimeError('');
        return true;
    };

    const handleTimeChange = (time: string) => {
        setBookingData({ ...bookingData, time });
        validateTime(time);
    };

    const submitBooking = async () => {
        if (!selectedDoctor || !bookingData.date || !bookingData.time) {
            return;
        }

        // Validate time is within allowed range
        if (!validateTime(bookingData.time)) {
            return;
        }

        setIsBooking(true);
        try {
            const response = await fetch(`${BACKEND_BASE}/api/appointments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: user.username,
                    patientName: `${user.firstName} ${user.lastName}`,
                    doctorId: selectedDoctor.id,
                    doctorName: `Dr. ${selectedDoctor.firstName} ${selectedDoctor.lastName}`,
                    date: bookingData.date,
                    time: bookingData.time,
                    reason: bookingData.reason,
                    status: 'pending',
                    createdAt: new Date().toISOString()
                })
            });

            const data = await response.json();

            if (data.success) {
                setBookingSuccess(true);
                setTimeout(() => {
                    setShowBookingDialog(false);
                    setBookingSuccess(false);
                }, 2000);
            }
        } catch (error) {
            console.error('Error booking appointment:', error);
        } finally {
            setIsBooking(false);
        }
    };

    if (loading) {
        return (
            <Card className="shadow-lg bg-white/80 backdrop-blur-sm border-blue-100">
                <CardContent className="p-8 flex items-center justify-center">
                    <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
                </CardContent>
            </Card>
        );
    }

    if (doctors.length === 0) {
        return null;
    }

    return (
        <>
            <Card className="shadow-lg bg-white/80 backdrop-blur-sm border-blue-100">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-slate-900">
                        <Stethoscope className="h-5 w-5 text-blue-600" />
                        Recommended Specialists
                    </CardTitle>
                    <CardDescription>
                        Based on your results, we recommend consulting with these specialists
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
                        {doctors.map((doctor) => (
                            <Card key={doctor.id} className="border-blue-100 hover:shadow-md transition-shadow">
                                <CardContent className="p-4">
                                    <div className="flex items-start gap-3 mb-3">
                                        <Avatar className="h-12 w-12 border-2 border-blue-100">
                                            <AvatarImage src={doctor.imageUrl} />
                                            <AvatarFallback className="bg-blue-100 text-blue-700">
                                                <UserCircle className="h-6 w-6" />
                                            </AvatarFallback>
                                        </Avatar>
                                        <div className="flex-1 min-w-0">
                                            <h4 className="font-semibold text-slate-900 truncate">
                                                Dr. {doctor.firstName} {doctor.lastName}
                                            </h4>
                                            <p className="text-xs text-slate-600 flex items-center gap-1 mt-1">
                                                <MapPin className="h-3 w-3" />
                                                {doctor.hospital}
                                            </p>
                                        </div>
                                    </div>

                                    <div className="space-y-2 mb-3">
                                        <div className="flex flex-wrap gap-1">
                                            {doctor.specialties.slice(0, 2).map((specialty, idx) => (
                                                <Badge key={idx} variant="secondary" className="text-xs bg-blue-50 text-blue-700 border-blue-100">
                                                    {specialty}
                                                </Badge>
                                            ))}
                                        </div>
                                        <p className="text-xs text-slate-600">
                                            {doctor.experience} years experience
                                        </p>
                                        <p className="text-sm font-semibold text-blue-600">
                                            ${doctor.consultationFee} consultation
                                        </p>
                                    </div>

                                    <div className="grid grid-cols-2 gap-2">
                                        <Button
                                            type="button"
                                            onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                handleViewProfile(doctor);
                                            }}
                                            variant="outline"
                                            className="w-full border-blue-200 text-blue-700 hover:bg-blue-50 text-sm cursor-pointer"
                                        >
                                            <User className="h-4 w-4 mr-2" />
                                            View Profile
                                        </Button>
                                        <Button
                                            type="button"
                                            onClick={(e) => {
                                                e.preventDefault();
                                                e.stopPropagation();
                                                handleBookAppointment(doctor);
                                            }}
                                            className="w-full bg-blue-600 hover:bg-blue-700 text-white text-sm cursor-pointer"
                                        >
                                            <Calendar className="h-4 w-4 mr-2" />
                                            Book
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Booking Dialog */}
            <Dialog open={showBookingDialog} onOpenChange={setShowBookingDialog}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <DialogTitle className="text-slate-900">Book Appointment</DialogTitle>
                        <DialogDescription>
                            {selectedDoctor && `Schedule a consultation with Dr. ${selectedDoctor.firstName} ${selectedDoctor.lastName}`}
                        </DialogDescription>
                    </DialogHeader>

                    {bookingSuccess ? (
                        <div className="py-8 text-center">
                            <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
                            <h3 className="text-lg font-semibold text-slate-900 mb-2">Appointment Booked!</h3>
                            <p className="text-sm text-slate-600">Your appointment request has been submitted successfully.</p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div>
                                <Label htmlFor="date" className="text-slate-700">Preferred Date</Label>
                                <Input
                                    id="date"
                                    type="date"
                                    value={bookingData.date}
                                    onChange={(e) => setBookingData({ ...bookingData, date: e.target.value })}
                                    min={new Date().toISOString().split('T')[0]}
                                    className="border-blue-100 focus:border-blue-400"
                                />
                            </div>

                            <div>
                                <Label htmlFor="time" className="text-slate-700">Preferred Time (10 AM - 4 PM)</Label>
                                <Select value={bookingData.time} onValueChange={handleTimeChange}>
                                    <SelectTrigger className="border-blue-100 focus:border-blue-400">
                                        <SelectValue placeholder="Select a time slot" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {generateTimeSlots().map((slot) => (
                                            <SelectItem key={slot} value={slot}>
                                                {new Date(`2000-01-01T${slot}`).toLocaleTimeString('en-US', {
                                                    hour: 'numeric',
                                                    minute: '2-digit',
                                                    hour12: true
                                                })}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                                {timeError && (
                                    <Alert variant="destructive" className="mt-2">
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertDescription>{timeError}</AlertDescription>
                                    </Alert>
                                )}
                            </div>

                            <div>
                                <Label htmlFor="reason" className="text-slate-700">Reason for Visit</Label>
                                <Textarea
                                    id="reason"
                                    value={bookingData.reason}
                                    onChange={(e) => setBookingData({ ...bookingData, reason: e.target.value })}
                                    rows={3}
                                    className="border-blue-100 focus:border-blue-400"
                                />
                            </div>

                            <div className="flex gap-2 pt-4">
                                <Button
                                    variant="outline"
                                    onClick={() => setShowBookingDialog(false)}
                                    className="flex-1 border-blue-200 text-blue-700 hover:bg-blue-50"
                                >
                                    Cancel
                                </Button>
                                <Button
                                    onClick={submitBooking}
                                    disabled={isBooking || !bookingData.date || !bookingData.time}
                                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                                >
                                    {isBooking ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Booking...
                                        </>
                                    ) : (
                                        'Confirm Booking'
                                    )}
                                </Button>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>

            {/* Doctor Profile Dialog */}
            <Dialog open={showProfileDialog} onOpenChange={setShowProfileDialog}>
                <DialogContent className="sm:max-w-2xl">
                    <DialogHeader>
                        <DialogTitle className="text-slate-900">Doctor Profile</DialogTitle>
                        <DialogDescription>
                            Detailed information about the specialist
                        </DialogDescription>
                    </DialogHeader>

                    {selectedDoctor && (
                        <div className="space-y-6">
                            {/* Doctor Header */}
                            <div className="flex items-start gap-4 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                                <Avatar className="h-20 w-20 border-4 border-white shadow-md">
                                    <AvatarImage src={selectedDoctor.imageUrl} />
                                    <AvatarFallback className="bg-blue-100 text-blue-700 text-2xl">
                                        <UserCircle className="h-10 w-10" />
                                    </AvatarFallback>
                                </Avatar>
                                <div className="flex-1">
                                    <h3 className="text-2xl font-bold text-slate-900">
                                        Dr. {selectedDoctor.firstName} {selectedDoctor.lastName}
                                    </h3>
                                    <div className="flex flex-wrap gap-2 mt-2">
                                        {selectedDoctor.specialties.map((specialty, idx) => (
                                            <Badge key={idx} className="bg-blue-600 text-white">
                                                {specialty}
                                            </Badge>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Doctor Details */}
                            <div className="grid md:grid-cols-2 gap-4">
                                <div className="space-y-3">
                                    <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                                        <MapPin className="h-5 w-5 text-blue-600 mt-0.5" />
                                        <div>
                                            <p className="text-sm font-semibold text-slate-700">Hospital</p>
                                            <p className="text-sm text-slate-600">{selectedDoctor.hospital}</p>
                                            <p className="text-xs text-slate-500 mt-1">{selectedDoctor.location}</p>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                                        <Stethoscope className="h-5 w-5 text-blue-600 mt-0.5" />
                                        <div>
                                            <p className="text-sm font-semibold text-slate-700">Experience</p>
                                            <p className="text-sm text-slate-600">{selectedDoctor.experience} years</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-3">
                                    <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                                        <Clock className="h-5 w-5 text-blue-600 mt-0.5" />
                                        <div>
                                            <p className="text-sm font-semibold text-slate-700">Availability</p>
                                            <p className="text-sm text-slate-600">Mon - Sat</p>
                                            <p className="text-xs text-slate-500 mt-1">10:00 AM - 4:00 PM</p>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                                        <Calendar className="h-5 w-5 text-blue-600 mt-0.5" />
                                        <div>
                                            <p className="text-sm font-semibold text-slate-700">Consultation Fee</p>
                                            <p className="text-lg font-bold text-blue-600">${selectedDoctor.consultationFee}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Specializations */}
                            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                                <h4 className="font-semibold text-slate-900 mb-2">Specializations</h4>
                                <div className="flex flex-wrap gap-2">
                                    {selectedDoctor.specialties.map((specialty, idx) => (
                                        <Badge key={idx} variant="outline" className="border-blue-300 text-blue-700">
                                            {specialty}
                                        </Badge>
                                    ))}
                                </div>
                            </div>

                            {/* Action Buttons */}
                            <div className="flex gap-3 pt-4 border-t">
                                <Button
                                    variant="outline"
                                    onClick={() => setShowProfileDialog(false)}
                                    className="flex-1 border-blue-200 text-blue-700 hover:bg-blue-50"
                                >
                                    Close
                                </Button>
                                <Button
                                    onClick={() => {
                                        setShowProfileDialog(false);
                                        handleBookAppointment(selectedDoctor);
                                    }}
                                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                                >
                                    <Calendar className="h-4 w-4 mr-2" />
                                    Book Appointment
                                </Button>
                            </div>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </>
    );
}
