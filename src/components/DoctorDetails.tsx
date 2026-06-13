import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import {
  MapPin,
  Star,
  Phone,
  Mail,
  Globe,
  Award,
  BookOpen,
  Calendar,
  Clock,
  DollarSign,
  Languages,
  GraduationCap,
  Building,
  Users,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { type Doctor } from '../utils/mockDoctorsData';
import { BACKEND_BASE } from '../utils/config';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './ui/dialog';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Alert, AlertDescription } from './ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';

interface DoctorDetailsProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
  };
  doctorId: string;
  onNavigateBack: () => void;
  onSpecialtyFilter: (specialty: string) => void;
}

export function DoctorDetails({ user, doctorId, onNavigateBack, onSpecialtyFilter }: DoctorDetailsProps) {
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showBookingDialog, setShowBookingDialog] = useState(false);
  const [bookingData, setBookingData] = useState({
    date: '',
    time: '',
    reason: ''
  });
  const [isBooking, setIsBooking] = useState(false);
  const [bookingSuccess, setBookingSuccess] = useState(false);
  const [timeError, setTimeError] = useState<string>('');

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
    if (!doctor || !bookingData.date || !bookingData.time) {
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
          doctorId: doctor.id,
          doctorName: `Dr. ${doctor.firstName} ${doctor.lastName}`,
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
          setBookingData({ date: '', time: '', reason: '' });
        }, 2000);
      }
    } catch (error) {
      console.error('Error booking appointment:', error);
    } finally {
      setIsBooking(false);
    }
  };

  useEffect(() => {
    const fetchDoctor = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const response = await fetch(`${BACKEND_BASE}/api/doctors/${doctorId}`);

        if (!response.ok) {
          throw new Error('Failed to fetch doctor details');
        }

        const data = await response.json();
        console.log('Fetched doctor details:', data);

        if (data.success && data.doctor) {
          setDoctor(data.doctor);
        } else {
          setError('Doctor not found');
        }
      } catch (error) {
        console.error('Error fetching doctor:', error);
        setError('Failed to load doctor details');
      } finally {
        setIsLoading(false);
      }
    };

    if (doctorId) {
      fetchDoctor();
    }
  }, [doctorId]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 pt-16 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 mx-auto mb-4 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          <h3 className="text-lg font-medium text-gray-900">Loading doctor profile...</h3>
        </div>
      </div>
    );
  }

  if (error || !doctor) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4 pt-16">
        <div className="max-w-4xl mx-auto">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center py-8">
                <h3 className="text-lg font-medium text-gray-900 mb-2">{error || 'Doctor not found'}</h3>
                <Button onClick={onNavigateBack}>Back to Specialists</Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  const defaultDoctorImage = "https://images.unsplash.com/photo-1758691461516-7e716e0ca135?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBkb2N0b3IlMjBwb3J0cmFpdCUyMG1lZGljYWx8ZW58MXx8fHwxNzU4OTQzMTUyfDA&ixlib=rb-4.1.0&q=80&w=400";

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-16">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <Button onClick={onNavigateBack} variant="outline" className="w-full sm:w-auto border-blue-200 text-blue-700 hover:bg-blue-50">
            ← Back to Specialists
          </Button>
          <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
            <Button variant="outline" onClick={() => setShowBookingDialog(true)} className="flex-1 sm:flex-none border-blue-200 text-blue-700 hover:bg-blue-50">
              <Calendar className="w-4 h-4 mr-2" />
              Schedule Consultation
            </Button>
            <Button className="flex-1 sm:flex-none bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md">
              <Mail className="w-4 h-4 mr-2" />
              Contact Doctor
            </Button>
          </div>
        </div>

        {/* Main Profile Card */}
        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-lg">
          <CardContent className="pt-6">
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Doctor Image and Basic Info */}
              <div className="lg:w-80 space-y-4">
                <div className="text-center">
                  <Avatar className="w-32 h-32 mx-auto mb-4 border-4 border-blue-100 ring-2 ring-blue-50">
                    <AvatarImage
                      src={doctor.imageUrl || defaultDoctorImage}
                      alt={`${doctor.firstName} ${doctor.lastName}`}
                    />
                    <AvatarFallback className="text-2xl bg-blue-100 text-blue-700">
                      {doctor.firstName[0]}{doctor.lastName[0]}
                    </AvatarFallback>
                  </Avatar>
                  <h1 className="text-2xl font-bold text-slate-900 mb-2">
                    Dr. {doctor.firstName} {doctor.lastName}
                  </h1>
                  <p className="text-lg text-slate-600 mb-2 font-medium">{doctor.position}</p>

                  <div className="flex items-center justify-center gap-2 mb-4">
                    <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                    <span className="text-lg font-medium text-slate-900">{doctor.rating}</span>
                    <span className="text-slate-500">({Math.floor(Math.random() * 200 + 50)} reviews)</span>
                  </div>

                  <Badge className="mb-4 text-lg px-3 py-1 bg-green-100 text-green-800 hover:bg-green-200 border-green-200">
                    <DollarSign className="w-4 h-4 mr-1" />
                    ${doctor.consultationFee} consultation
                  </Badge>
                </div>

                {/* Quick Actions */}
                <div className="space-y-2">
                  <Button onClick={() => setShowBookingDialog(true)} className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md" size="lg">
                    <Calendar className="w-5 h-5 mr-2" />
                    Book Appointment
                  </Button>
                  <Button variant="outline" className="w-full border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Phone className="w-4 h-4 mr-2" />
                    Call Now
                  </Button>
                  <Button variant="outline" className="w-full border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Mail className="w-4 h-4 mr-2" />
                    Send Message
                  </Button>
                </div>
              </div>

              {/* Detailed Information */}
              <div className="flex-1 space-y-6">
                {/* Biography */}
                {doctor.biography && (
                  <div>
                    <h3 className="text-lg font-semibold mb-3 text-slate-900">About Dr. {doctor.lastName}</h3>
                    <p className="text-slate-700 leading-relaxed">{doctor.biography}</p>
                  </div>
                )}

                {/* Education & Credentials */}
                <div>
                  <h3 className="text-lg font-semibold mb-3 flex items-center text-slate-900">
                    <GraduationCap className="w-5 h-5 mr-2 text-blue-600" />
                    Education & Credentials
                  </h3>
                  <div className="space-y-2">
                    {doctor.degrees.map((degree, index) => (
                      <div key={index} className="flex items-center gap-2">
                        <Badge variant="outline" className="border-blue-200 text-blue-700 bg-blue-50">{degree}</Badge>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Hospital & Location */}
                <div>
                  <h3 className="text-lg font-semibold mb-3 flex items-center text-slate-900">
                    <Building className="w-5 h-5 mr-2 text-blue-600" />
                    Hospital & Location
                  </h3>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Building className="w-4 h-4 text-slate-500" />
                      <span className="font-medium text-slate-900">{doctor.hospital}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <MapPin className="w-4 h-4 text-slate-500" />
                      <span className="text-slate-700">{doctor.location}</span>
                    </div>
                  </div>
                </div>

                {/* Contact Information */}
                <div>
                  <h3 className="text-lg font-semibold mb-3 text-slate-900">Contact Information</h3>
                  <div className="space-y-2">
                    <div className="flex items-center gap-2">
                      <Mail className="w-4 h-4 text-slate-500" />
                      <a href={`mailto:${doctor.email}`} className="text-blue-600 hover:underline hover:text-blue-800 transition-colors">
                        {doctor.email}
                      </a>
                    </div>
                    <div className="flex items-center gap-2">
                      <Phone className="w-4 h-4 text-slate-500" />
                      <a href={`tel:${doctor.phone}`} className="text-blue-600 hover:underline hover:text-blue-800 transition-colors">
                        {doctor.phone}
                      </a>
                    </div>
                    {doctor.website && (
                      <div className="flex items-center gap-2">
                        <Globe className="w-4 h-4 text-slate-500" />
                        <a href={doctor.website} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline hover:text-blue-800 transition-colors">
                          Professional Website
                        </a>
                      </div>
                    )}
                    {doctor.linkedIn && (
                      <div className="flex items-center gap-2">
                        <Users className="w-4 h-4 text-slate-500" />
                        <a href={doctor.linkedIn} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline hover:text-blue-800 transition-colors">
                          LinkedIn Profile
                        </a>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Additional Information Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Specialties & Expertise */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center text-slate-900">
                <Award className="w-5 h-5 mr-2 text-blue-600" />
                Areas of Expertise
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {doctor.specialties.map((specialty, index) => (
                  <button
                    key={index}
                    onClick={() => onSpecialtyFilter(specialty)}
                    className="block w-full text-left px-3 py-2 bg-blue-50 text-blue-700 rounded-lg hover:bg-blue-100 transition-colors border border-blue-100"
                  >
                    {specialty}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Professional Details */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center text-slate-900">
                <Clock className="w-5 h-5 mr-2 text-blue-600" />
                Professional Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-600">Experience:</span>
                  <span className="font-medium text-slate-900">{doctor.experience} years</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-600">Age:</span>
                  <span className="font-medium text-slate-900">{doctor.age} years</span>
                </div>
                <div className="flex justify-between border-b border-slate-100 pb-2">
                  <span className="text-slate-600">Gender:</span>
                  <span className="font-medium text-slate-900">{doctor.gender}</span>
                </div>
                {doctor.publications && (
                  <div className="flex justify-between border-b border-slate-100 pb-2">
                    <span className="text-slate-600">Publications:</span>
                    <span className="font-medium text-slate-900">{doctor.publications}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-slate-600">Languages:</span>
                  <span className="font-medium text-slate-900">{doctor.languages.join(', ')}</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Awards & Recognition */}
          {doctor.awards && doctor.awards.length > 0 && (
            <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center text-slate-900">
                  <Award className="w-5 h-5 mr-2 text-yellow-500" />
                  Awards & Recognition
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {doctor.awards.map((award, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-yellow-50/50 rounded-lg border border-yellow-100">
                      <Award className="w-4 h-4 text-yellow-500" />
                      <span className="text-slate-700">{award}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Availability & Booking */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center text-slate-900">
                <Calendar className="w-5 h-5 mr-2 text-blue-600" />
                Availability & Booking
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2 text-slate-900">Available Times:</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-slate-600">
                    <div className="bg-slate-50 p-2 rounded">Monday: 9:00 AM - 5:00 PM</div>
                    <div className="bg-slate-50 p-2 rounded">Tuesday: 9:00 AM - 5:00 PM</div>
                    <div className="bg-slate-50 p-2 rounded">Wednesday: 9:00 AM - 5:00 PM</div>
                    <div className="bg-slate-50 p-2 rounded">Thursday: 9:00 AM - 5:00 PM</div>
                    <div className="bg-slate-50 p-2 rounded">Friday: 9:00 AM - 3:00 PM</div>
                    <div className="bg-slate-50 p-2 rounded">Weekend: By appointment</div>
                  </div>
                </div>
                <div className="border-t border-slate-100 pt-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-slate-600">Consultation Fee:</span>
                    <span className="text-xl font-bold text-green-600">${doctor.consultationFee}</span>
                  </div>
                  <p className="text-sm text-slate-500">
                    Online consultations available at reduced rates
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Call to Action */}
        <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200 shadow-md">
          <CardContent className="pt-6">
            <div className="text-center">
              <h3 className="text-xl font-semibold text-blue-900 mb-2">
                Ready to Schedule a Consultation?
              </h3>
              <p className="text-blue-700 mb-4">
                Book an appointment with Dr. {doctor.lastName} to discuss your health concerns
              </p>
              <div className="flex justify-center gap-4">
                <Button onClick={() => setShowBookingDialog(true)} size="lg" className="bg-blue-600 hover:bg-blue-700 text-white shadow-md">
                  <Calendar className="w-5 h-5 mr-2" />
                  Schedule Appointment
                </Button>
                <Button variant="outline" size="lg" className="border-blue-200 text-blue-700 hover:bg-blue-100 bg-white">
                  <Phone className="w-5 h-5 mr-2" />
                  Call Directly
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Booking Dialog */}
      <Dialog open={showBookingDialog} onOpenChange={setShowBookingDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-slate-900">Book Appointment</DialogTitle>
            <DialogDescription>
              {doctor && `Schedule a consultation with Dr. ${doctor.firstName} ${doctor.lastName}`}
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
                  placeholder="Briefly describe your symptoms or reason for consultation..."
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
    </div>
  );
}