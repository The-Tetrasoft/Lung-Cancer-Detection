import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import {
  Search,
  Plus,
  Edit,
  Trash2,
  Users,
  Building,
  MapPin,
  Star,
  AlertTriangle,
  ArrowLeft
} from 'lucide-react';
import { mockDoctors, searchDoctors, type Doctor } from '../utils/mockDoctorsData';
import { Alert, AlertDescription } from './ui/alert';
import { db } from '../firebase-client';
import { collection, getDocs, deleteDoc, doc } from 'firebase/firestore';

interface ManageDoctorsProps {
  onAddDoctor: () => void;
  onEditDoctor: (doctorId: string) => void;
  onNavigateBack: () => void;
}

export function ManageDoctors({ onAddDoctor, onEditDoctor, onNavigateBack }: ManageDoctorsProps) {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [filteredDoctors, setFilteredDoctors] = useState<Doctor[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch doctors from Firestore
  const fetchDoctors = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const doctorsRef = collection(db, 'doctors');
      const snapshot = await getDocs(doctorsRef);

      const fetchedDoctors: Doctor[] = snapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          firstName: data.firstName || '',
          lastName: data.lastName || '',
          specialties: data.specialties || [],
          hospital: data.hospital || '',
          location: data.location || '',
          rating: data.rating || 0,
          experience: data.experience || 0,
          consultationFee: data.consultationFee || 0,
          imageUrl: data.imageUrl || '',
          position: data.position || 'Specialist'
        } as Doctor;
      });

      setDoctors(fetchedDoctors);
    } catch (error) {
      console.error('Error fetching doctors:', error);
      setError('Failed to load doctors from database.');
      setDoctors([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDoctors();
  }, []);

  useEffect(() => {
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      const results = doctors.filter(doctor => {
        const fullName = `${doctor.firstName} ${doctor.lastName}`.toLowerCase();
        const hospital = doctor.hospital?.toLowerCase() || '';
        const specialtiesStr = doctor.specialties?.join(' ').toLowerCase() || '';

        return fullName.includes(term) ||
          hospital.includes(term) ||
          specialtiesStr.includes(term);
      });
      setFilteredDoctors(results);
    } else {
      setFilteredDoctors(doctors);
    }
  }, [searchTerm, doctors]);

  const handleDeleteDoctor = async (doctorId: string) => {
    try {
      await deleteDoc(doc(db, 'doctors', doctorId));

      // Refresh the doctors list
      await fetchDoctors();
      setDeleteConfirm(null);
      alert('Doctor deleted successfully!');
    } catch (error) {
      console.error('Error deleting doctor:', error);
      alert('Failed to delete doctor. Please try again.');
    }
  };

  const defaultDoctorImage = "https://images.unsplash.com/photo-1758691461516-7e716e0ca135?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBkb2N0b3IlMjBwb3J0cmFpdCUyMG1lZGljYWx8ZW58MXx8fHwxNzU4OTQzMTUyfDA&ixlib=rb-4.1.0&q=80&w=400";

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-16">
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
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">Manage Doctors</h1>
            <p className="text-slate-600">Add, edit, and manage doctor profiles</p>
          </div>
        </div>

        {/* Action Bar */}
        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
              <div className="flex-1 w-full md:max-w-md">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                  <Input
                    placeholder="Search by name, hospital, or specialty..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 border-blue-100 focus:border-blue-400"
                  />
                </div>
              </div>
              <Button onClick={onAddDoctor} className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md">
                <Plus className="w-4 h-4" />
                Add New Doctor
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Statistics */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="pt-6 flex flex-col items-center justify-center text-center">
              <div className="text-2xl font-bold text-blue-600">{doctors.length}</div>
              <div className="text-sm text-slate-600">Total Doctors</div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-teal-100 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="pt-6 flex flex-col items-center justify-center text-center">
              <div className="text-2xl font-bold text-teal-600">
                {new Set(doctors.flatMap(d => d.specialties)).size}
              </div>
              <div className="text-sm text-slate-600">Specialties</div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-indigo-100 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="pt-6 flex flex-col items-center justify-center text-center">
              <div className="text-2xl font-bold text-indigo-600">
                {new Set(doctors.map(d => d.hospital)).size}
              </div>
              <div className="text-sm text-slate-600">Hospitals</div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-amber-100 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="pt-6 flex flex-col items-center justify-center text-center">
              <div className="text-2xl font-bold text-amber-600">
                {(doctors.reduce((sum, d) => sum + d.rating, 0) / doctors.length).toFixed(1)}
              </div>
              <div className="text-sm text-slate-600">Avg. Rating</div>
            </CardContent>
          </Card>
        </div>

        {/* Doctors Table */}
        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-slate-900">
              <Users className="w-5 h-5 text-blue-600" />
              Doctors Directory ({filteredDoctors.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="text-center py-12">
                <div className="w-12 h-12 mx-auto mb-4 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                <h3 className="text-lg font-medium text-slate-900 mb-2">Loading doctors...</h3>
                <p className="text-slate-600">Please wait while we fetch the doctor list.</p>
              </div>
            ) : error ? (
              <div className="text-center py-8">
                <AlertTriangle className="w-12 h-12 mx-auto text-red-400 mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">Error Loading Doctors</h3>
                <p className="text-slate-600 mb-4">{error}</p>
                <Button onClick={() => fetchDoctors()} variant="destructive">Retry</Button>
              </div>
            ) : filteredDoctors.length === 0 ? (
              <div className="text-center py-8">
                <Users className="w-12 h-12 mx-auto text-slate-400 mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">No doctors found</h3>
                <p className="text-slate-600">Try adjusting your search criteria or add a new doctor.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredDoctors.map((doctor) => (
                  <div key={doctor.id} className="border border-blue-50 rounded-lg p-4 hover:bg-blue-50/50 transition-colors bg-white/50">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                      {/* Doctor Info Column */}
                      <div className="flex items-center gap-4 flex-1 w-full">
                        <Avatar className="w-16 h-16 border-2 border-blue-100 ring-2 ring-blue-50 flex-shrink-0">
                          <AvatarImage
                            src={doctor.imageUrl || defaultDoctorImage}
                            alt={`${doctor.firstName} ${doctor.lastName}`}
                          />
                          <AvatarFallback className="bg-blue-100 text-blue-700">
                            {doctor.firstName[0]}{doctor.lastName[0]}
                          </AvatarFallback>
                        </Avatar>

                        <div className="flex-1 min-w-0">
                          <h3 className="font-bold text-lg text-slate-900 truncate">
                            Dr. {doctor.firstName} {doctor.lastName}
                          </h3>
                          <div className="space-y-1">
                            <div className="flex items-center gap-2 text-sm text-slate-600">
                              <Building className="w-4 h-4 text-slate-400 flex-shrink-0" />
                              <span className="font-medium truncate">{doctor.hospital}</span>
                            </div>
                            <div className="flex flex-wrap items-center gap-2 text-sm text-slate-600">
                              <span>{doctor.position}</span>
                              <span className="text-slate-300 hidden sm:inline">•</span>
                              <span className="text-xs sm:text-sm">{doctor.experience} years exp.</span>
                            </div>
                            <div className="flex items-center gap-2 text-sm text-slate-600">
                              <MapPin className="w-4 h-4 text-slate-400 flex-shrink-0" />
                              <span className="truncate">{doctor.location}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Star className="w-4 h-4 fill-yellow-400 text-yellow-400 flex-shrink-0" />
                              <span className="text-sm font-medium text-slate-700">{doctor.rating}</span>
                              <Badge variant="outline" className="text-xs border-green-200 text-green-700 bg-green-50">
                                ${doctor.consultationFee}
                              </Badge>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Specialties Column */}
                      <div className="w-full sm:w-auto sm:flex-1 px-0 sm:px-4">
                        <p className="text-sm font-medium text-slate-700 mb-2">Specialties:</p>
                        <div className="flex flex-wrap gap-1">
                          {doctor.specialties.slice(0, 3).map((specialty, index) => (
                            <Badge key={index} variant="secondary" className="text-xs bg-blue-50 text-blue-700 hover:bg-blue-100 border border-blue-100">
                              {specialty}
                            </Badge>
                          ))}
                          {doctor.specialties.length > 3 && (
                            <Badge variant="outline" className="text-xs text-slate-500 border-slate-200">
                              +{doctor.specialties.length - 3} more
                            </Badge>
                          )}
                        </div>
                      </div>

                      {/* Actions Column */}
                      <div className="flex gap-2 w-full sm:w-auto mt-2 sm:mt-0">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onEditDoctor(doctor.id)}
                          className="flex-1 sm:flex-none flex items-center justify-center gap-1 border-blue-200 text-blue-700 hover:bg-blue-50"
                        >
                          <Edit className="w-4 h-4" />
                          Edit
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setDeleteConfirm(doctor.id)}
                          className="flex-1 sm:flex-none flex items-center justify-center gap-1 text-red-600 hover:text-red-700 hover:bg-red-50 border-red-200"
                        >
                          <Trash2 className="w-4 h-4" />
                          Delete
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Delete Confirmation Dialog */}
        {deleteConfirm && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
            <Card className="w-full max-w-md m-4 bg-white/95 border-red-100 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-600">
                  <AlertTriangle className="w-5 h-5" />
                  Confirm Deletion
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Alert className="bg-red-50 border-red-100 text-red-800">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <AlertDescription>
                    Are you sure you want to delete this doctor's profile? This action cannot be undone and will remove all associated data.
                  </AlertDescription>
                </Alert>
                <div className="flex gap-3 mt-6">
                  <Button
                    variant="outline"
                    onClick={() => setDeleteConfirm(null)}
                    className="flex-1 border-slate-200 hover:bg-slate-50"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => handleDeleteDoctor(deleteConfirm)}
                    className="flex-1 bg-red-600 hover:bg-red-700"
                  >
                    Delete Doctor
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}