import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Search, MapPin, Star, Phone, Mail, Globe, Filter, Users } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { mockDoctors, specialties, getDoctorsBySpecialty, searchDoctors, type Doctor } from '../utils/mockDoctorsData';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { BACKEND_BASE } from '../utils/config';
import { db } from '../firebase-client';
import { collection, getDocs } from 'firebase/firestore';

interface DoctorsListProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
  };
  selectedSpecialty: string | null;
  onDoctorSelect: (doctorId: string) => void;
  onSpecialtyFilter: (specialty: string) => void;
  onNavigateBack: () => void;
}

export function DoctorsList({
  user,
  selectedSpecialty,
  onDoctorSelect,
  onSpecialtyFilter,
  onNavigateBack
}: DoctorsListProps) {
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [filteredDoctors, setFilteredDoctors] = useState<Doctor[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState<string>('rating');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch doctors from Firestore
  useEffect(() => {
    const fetchDoctors = async () => {
      try {
        setIsLoading(true);
        setError(null);

        const doctorsRef = collection(db, 'doctors');
        const querySnapshot = await getDocs(doctorsRef);

        const fetchedDoctors: Doctor[] = querySnapshot.docs.map(doc => {
          const data = doc.data();
          return {
            id: doc.id,
            firstName: data.firstName || '',
            lastName: data.lastName || '',
            email: data.email || '',
            phone: data.phone || '',
            hospital: data.hospital || '',
            specialties: data.specialties || [],
            rating: data.rating || 0,
            experience: data.experience || 0,
            consultationFee: data.consultationFee || 0,
            imageUrl: data.imageUrl || '',
            position: data.position || '',
            degrees: data.degrees || [],
            location: data.location || '',
            about: data.about || '',
            availability: data.availability || {}
          } as Doctor;
        });

        console.log('Fetched doctors from Firestore:', fetchedDoctors);
        setDoctors(fetchedDoctors);

      } catch (error: any) {
        console.error('Error fetching doctors:', error);
        if (error.code === 'permission-denied') {
          setError('Access Denied: Please check Firestore Rules.');
        } else {
          setError('Failed to load doctors. Please try again later.');
        }
        setDoctors([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDoctors();
  }, []); // Run once on mount

  useEffect(() => {
    let filtered = [...doctors];

    // Apply specialty filter
    if (selectedSpecialty) {
      filtered = filtered.filter(doctor =>
        doctor.specialties?.some(s =>
          s.toLowerCase().includes(selectedSpecialty.toLowerCase())
        )
      );
    }

    // Apply search filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(doctor => {
        const fullName = `${doctor.firstName} ${doctor.lastName}`.toLowerCase();
        const hospital = doctor.hospital?.toLowerCase() || '';
        const specialtiesStr = doctor.specialties?.join(' ').toLowerCase() || '';

        return fullName.includes(term) ||
          hospital.includes(term) ||
          specialtiesStr.includes(term);
      });
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'rating':
          return (b.rating || 0) - (a.rating || 0);
        case 'experience':
          return (b.experience || 0) - (a.experience || 0);
        case 'name':
          return `${a.firstName} ${a.lastName}`.localeCompare(`${b.firstName} ${b.lastName}`);
        case 'fee':
          return (a.consultationFee || 0) - (b.consultationFee || 0);
        default:
          return 0;
      }
    });

    setFilteredDoctors(filtered);
  }, [searchTerm, selectedSpecialty, sortBy, doctors]);

  const clearFilters = () => {
    setSearchTerm('');
    onSpecialtyFilter('');
    setSortBy('rating');
  };

  const defaultDoctorImage = "https://images.unsplash.com/photo-1758691461516-7e716e0ca135?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBkb2N0b3IlMjBwb3J0cmFpdCUyMG1lZGljYWx8ZW58MXx8fHwxNzU4OTQzMTUyfDA&ixlib=rb-4.1.0&q=80&w=400";

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-16">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">Find Specialists</h1>
            <p className="text-slate-600">Connect with qualified doctors and specialists</p>
          </div>
          <Button onClick={onNavigateBack} variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50">
            Back to Dashboard
          </Button>
        </div>

        {/* Search and Filters */}
        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
          <CardContent className="pt-6">
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                  <Input
                    placeholder="Search by doctor name, hospital, or specialty..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 border-blue-100 focus:border-blue-400"
                  />
                </div>
              </div>

              <Select value={selectedSpecialty || 'all'} onValueChange={(value) => onSpecialtyFilter(value === 'all' ? '' : value)}>
                <SelectTrigger className="w-full max-w-sm border-blue-100">
                  <Filter className="w-4 h-4 mr-2 text-slate-500" />
                  <SelectValue placeholder="Filter by Specialty" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Specialties</SelectItem>
                  {specialties.map((specialty) => (
                    <SelectItem key={specialty} value={specialty}>
                      {specialty}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-full max-w-xs border-blue-100">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="rating">Rating</SelectItem>
                  <SelectItem value="experience">Experience</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="fee">Fee (Low to High)</SelectItem>
                </SelectContent>
              </Select>

              {(searchTerm || selectedSpecialty) && (
                <Button variant="outline" onClick={clearFilters} className="border-blue-200 text-blue-700 hover:bg-blue-50">
                  Clear Filters
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Quick Specialty Filters */}
        <div className="flex flex-wrap gap-2">
          {['Lung Cancer', 'Pulmonary Medicine', 'Thoracic Surgery', 'Medical Oncology', 'Radiology'].map((specialty) => (
            <Button
              key={specialty}
              variant={selectedSpecialty === specialty ? "default" : "outline"}
              size="sm"
              onClick={() => onSpecialtyFilter(specialty)}
              className={selectedSpecialty === specialty
                ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white border-0"
                : "bg-white/50 border-blue-200 text-blue-700 hover:bg-blue-50"}
            >
              {specialty}
            </Button>
          ))}
        </div>

        {/* Results Count */}
        <div className="flex items-center gap-2 text-slate-600">
          <Users className="w-4 h-4 text-slate-400" />
          <span>{filteredDoctors.length} specialists found</span>
          {selectedSpecialty && (
            <Badge variant="secondary" className="bg-blue-100 text-blue-700 hover:bg-blue-200">{selectedSpecialty}</Badge>
          )}
        </div>

        {/* Doctors Grid */}
        {isLoading ? (
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
            <CardContent className="pt-6">
              <div className="text-center py-12">
                <div className="w-12 h-12 mx-auto mb-4 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                <h3 className="text-lg font-medium text-slate-900 mb-2">Loading specialists...</h3>
                <p className="text-slate-600">Please wait while we fetch the latest doctor information.</p>
              </div>
            </CardContent>
          </Card>
        ) : error ? (
          <Card className="bg-white/80 backdrop-blur-sm border-red-100">
            <CardContent className="pt-6">
              <div className="text-center py-8">
                <Users className="w-12 h-12 mx-auto text-red-400 mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">Error Loading Doctors</h3>
                <p className="text-slate-600 mb-4">{error}</p>
                <Button onClick={() => window.location.reload()} variant="destructive">Retry</Button>
              </div>
            </CardContent>
          </Card>
        ) : filteredDoctors.length === 0 ? (
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
            <CardContent className="pt-6">
              <div className="text-center py-8">
                <Users className="w-12 h-12 mx-auto text-slate-400 mb-4" />
                <h3 className="text-lg font-medium text-slate-900 mb-2">No specialists found</h3>
                <p className="text-slate-600">
                  {doctors.length === 0
                    ? 'No doctors have been added yet. Please check back later or contact an administrator.'
                    : 'Try adjusting your search criteria or clear filters.'}
                </p>
              </div>
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredDoctors.map((doctor) => (
              <Card key={doctor.id} className="hover:shadow-xl transition-all cursor-pointer bg-white/90 backdrop-blur-sm border-blue-50 hover:border-blue-200 group">
                <CardContent className="pt-6">
                  <div className="flex gap-4">
                    {/* Doctor Image */}
                    <div className="flex-shrink-0">
                      <Avatar className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 border-2 border-blue-100 ring-2 ring-blue-50">
                        <AvatarImage
                          src={doctor.imageUrl || defaultDoctorImage}
                          alt={`${doctor.firstName} ${doctor.lastName}`}
                        />
                        <AvatarFallback className="bg-blue-100 text-blue-700">
                          {doctor.firstName[0]}{doctor.lastName[0]}
                        </AvatarFallback>
                      </Avatar>
                    </div>

                    {/* Doctor Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-col sm:flex-row items-start justify-between gap-4">
                        <div className="flex-1 w-full">
                          <h3 className="text-lg font-bold text-slate-900 mb-1 group-hover:text-blue-700 transition-colors">
                            Dr. {doctor.firstName} {doctor.lastName}
                          </h3>
                          <p className="text-slate-600 text-sm mb-1 font-medium">
                            {doctor.position}
                          </p>
                          <p className="text-slate-500 text-sm mb-2">
                            {doctor.degrees.join(', ')}
                          </p>
                          <p className="text-slate-600 text-sm font-medium mb-3 flex items-center gap-1">
                            <MapPin className="w-3 h-3 text-blue-500" />
                            {doctor.hospital}
                          </p>

                          <div className="space-y-1 text-sm text-slate-600 mb-3">
                            <div className="flex items-center gap-2">
                              <Mail className="w-3 h-3 text-slate-400" />
                              <span className="break-all">{doctor.email}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Phone className="w-3 h-3 text-slate-400" />
                              <span>{doctor.phone}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <MapPin className="w-3 h-3 text-slate-400" />
                              <span>{doctor.location}</span>
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-4 mb-3">
                            <div className="flex items-center gap-1 bg-yellow-50 px-2 py-0.5 rounded-full border border-yellow-100">
                              <Star className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                              <span className="text-sm font-medium text-yellow-700">{doctor.rating}</span>
                            </div>
                            <span className="text-sm text-slate-600">
                              {doctor.experience} years exp.
                            </span>
                            <span className="text-sm text-green-600 font-medium bg-green-50 px-2 py-0.5 rounded-full border border-green-100">
                              ${doctor.consultationFee}
                            </span>
                          </div>

                          <Button
                            onClick={() => onDoctorSelect(doctor.id)}
                            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md hover:shadow-lg transition-all"
                          >
                            View Profile
                          </Button>
                        </div>

                        {/* Specialties */}
                        <div className="w-full sm:w-auto sm:ml-4 sm:text-right min-w-0 sm:max-w-[18rem] md:max-w-[12rem]">
                          <p className="text-sm font-medium text-slate-700 mb-2">Research Areas</p>
                          <div className="flex flex-wrap sm:flex-col gap-2 sm:gap-1">
                            {doctor.specialties.slice(0, 4).map((specialty, index) => (
                              <button
                                key={index}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onSpecialtyFilter(specialty);
                                }}
                                className="block text-xs px-2 py-1 bg-blue-50 text-blue-700 rounded hover:bg-blue-100 transition-colors text-left sm:w-full border border-blue-100"
                              >
                                {specialty}
                              </button>
                            ))}
                            {doctor.specialties.length > 4 && (
                              <span className="text-xs text-slate-500">
                                +{doctor.specialties.length - 4} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}