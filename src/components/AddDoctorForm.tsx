import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import {
  Upload,
  X,
  Plus,
  User,
  Mail,
  Phone,
  MapPin,
  Building,
  GraduationCap,
  Award,
  Save,
  AlertCircle,
  Check
} from 'lucide-react';
import { Alert, AlertDescription } from './ui/alert';
import { getDoctorById, specialties, type Doctor } from '../utils/mockDoctorsData';

interface AddDoctorFormProps {
  doctorId: string | null; // null for new doctor, string for editing
  onSuccess: () => void;
  onCancel: () => void;
}

interface FormData {
  firstName: string;
  lastName: string;
  gender: 'Male' | 'Female' | 'Other' | '';
  age: string;
  hospital: string;
  position: string;
  degrees: string[];
  location: string;
  emails: string[];
  phones: string[];
  website: string;
  linkedIn: string;
  specialties: string[];
  biography: string;
  experience: string;
  languages: string[];
  awards: string[];
  consultationFee: string;
  imageFile: File | null;
  imagePreview: string;
}

export function AddDoctorForm({ doctorId, onSuccess, onCancel }: AddDoctorFormProps) {
  const isEditing = doctorId !== null;
  const [formData, setFormData] = useState<FormData>({
    firstName: '',
    lastName: '',
    gender: '',
    age: '',
    hospital: '',
    position: '',
    degrees: [],
    location: '',
    emails: [''],
    phones: [''],
    website: '',
    linkedIn: '',
    specialties: [],
    biography: '',
    experience: '',
    languages: [],
    awards: [],
    consultationFee: '',
    imageFile: null,
    imagePreview: ''
  });

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [newDegree, setNewDegree] = useState('');
  const [newSpecialty, setNewSpecialty] = useState('');
  const [newLanguage, setNewLanguage] = useState('');
  const [newAward, setNewAward] = useState('');

  // Load existing doctor data if editing
  useEffect(() => {
    if (isEditing && doctorId) {
      const doctor = getDoctorById(doctorId);
      if (doctor) {
        setFormData({
          firstName: doctor.firstName,
          lastName: doctor.lastName,
          gender: doctor.gender,
          age: doctor.age.toString(),
          hospital: doctor.hospital,
          position: doctor.position,
          degrees: doctor.degrees,
          location: doctor.location,
          emails: [doctor.email],
          phones: [doctor.phone],
          website: doctor.website || '',
          linkedIn: doctor.linkedIn || '',
          specialties: doctor.specialties,
          biography: doctor.biography || '',
          experience: doctor.experience.toString(),
          languages: doctor.languages,
          awards: doctor.awards || [],
          consultationFee: doctor.consultationFee.toString(),
          imageFile: null,
          imagePreview: doctor.imageUrl || ''
        });
      }
    }
  }, [isEditing, doctorId]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Required fields
    if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
    if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
    if (!formData.gender) newErrors.gender = 'Gender is required';
    if (!formData.age || isNaN(Number(formData.age)) || Number(formData.age) < 25 || Number(formData.age) > 80) {
      newErrors.age = 'Please enter a valid age between 25-80';
    }
    if (!formData.hospital.trim()) newErrors.hospital = 'Hospital is required';
    if (!formData.position.trim()) newErrors.position = 'Position is required';
    if (formData.degrees.length === 0) newErrors.degrees = 'At least one degree is required';
    if (!formData.location.trim()) newErrors.location = 'Location is required';
    if (!formData.experience || isNaN(Number(formData.experience)) || Number(formData.experience) < 0) {
      newErrors.experience = 'Please enter valid years of experience';
    }
    if (!formData.consultationFee || isNaN(Number(formData.consultationFee)) || Number(formData.consultationFee) < 0) {
      newErrors.consultationFee = 'Please enter a valid consultation fee';
    }

    // Email validation
    const validEmails = formData.emails.filter(email => email.trim());
    if (validEmails.length === 0) {
      newErrors.emails = 'At least one email is required';
    } else {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      const invalidEmails = validEmails.filter(email => !emailRegex.test(email));
      if (invalidEmails.length > 0) {
        newErrors.emails = 'Please enter valid email addresses';
      }
    }

    // Phone validation
    const validPhones = formData.phones.filter(phone => phone.trim());
    if (validPhones.length === 0) {
      newErrors.phones = 'At least one phone number is required';
    } else {
      const phoneRegex = /^\+?[\d\s\-\(\)]{10,}$/;
      const invalidPhones = validPhones.filter(phone => !phoneRegex.test(phone.replace(/\s/g, '')));
      if (invalidPhones.length > 0) {
        newErrors.phones = 'Please enter valid phone numbers';
      }
    }

    // Specialties validation
    if (formData.specialties.length === 0) {
      newErrors.specialties = 'At least one specialty is required';
    }

    // Languages validation
    if (formData.languages.length === 0) {
      newErrors.languages = 'At least one language is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      setErrors({ ...errors, image: 'Only JPEG, PNG, and WebP images are allowed' });
      return;
    }

    // Validate file size (max 5MB)
    const maxSize = 5 * 1024 * 1024; // 5MB
    if (file.size > maxSize) {
      setErrors({ ...errors, image: 'Image size must be less than 5MB' });
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setFormData(prev => ({
        ...prev,
        imageFile: file,
        imagePreview: e.target?.result as string
      }));
      setErrors(prev => ({ ...prev, image: '' }));
    };
    reader.readAsDataURL(file);
  };

  const addToArray = (field: keyof FormData, value: string) => {
    if (!value.trim()) return;

    const currentArray = formData[field] as string[];
    if (!currentArray.includes(value.trim())) {
      setFormData(prev => ({
        ...prev,
        [field]: [...currentArray, value.trim()]
      }));
    }
  };

  const removeFromArray = (field: keyof FormData, index: number) => {
    const currentArray = formData[field] as string[];
    setFormData(prev => ({
      ...prev,
      [field]: currentArray.filter((_, i) => i !== index)
    }));
  };

  const addEmail = () => {
    setFormData(prev => ({
      ...prev,
      emails: [...prev.emails, '']
    }));
  };

  const addPhone = () => {
    setFormData(prev => ({
      ...prev,
      phones: [...prev.phones, '']
    }));
  };

  const updateEmail = (index: number, value: string) => {
    const newEmails = [...formData.emails];
    newEmails[index] = value;
    setFormData(prev => ({ ...prev, emails: newEmails }));
  };

  const updatePhone = (index: number, value: string) => {
    const newPhones = [...formData.phones];
    newPhones[index] = value;
    setFormData(prev => ({ ...prev, phones: newPhones }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setErrors({}); // Clear previous errors

    try {
      // Prepare doctor data
      const doctorPayload = {
        firstName: formData.firstName,
        lastName: formData.lastName,
        gender: formData.gender,
        age: Number(formData.age),
        hospital: formData.hospital,
        position: formData.position,
        degrees: formData.degrees,
        location: formData.location,
        email: formData.emails.filter(e => e.trim())[0], // Primary email
        phone: formData.phones.filter(p => p.trim())[0], // Primary phone
        emails: formData.emails.filter(e => e.trim()),
        phones: formData.phones.filter(p => p.trim()),
        website: formData.website,
        linkedIn: formData.linkedIn,
        specialties: formData.specialties,
        biography: formData.biography,
        experience: Number(formData.experience),
        languages: formData.languages,
        awards: formData.awards,
        consultationFee: Number(formData.consultationFee),
        imageUrl: formData.imagePreview || '', // In production, upload to storage first
      };

      const BACKEND_BASE = import.meta.env.VITE_BACKEND_BASE ?? 'http://localhost:8000';

      console.log('Submitting doctor data:', doctorPayload);
      console.log('Backend URL:', BACKEND_BASE);

      if (isEditing && doctorId) {
        // Update existing doctor
        const response = await fetch(`${BACKEND_BASE}/api/doctors/${doctorId}`, {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(doctorPayload),
        });

        console.log('Update response status:', response.status);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ message: 'Failed to update doctor' }));
          console.error('Update error:', errorData);
          throw new Error(errorData.message || 'Failed to update doctor');
        }

        const data = await response.json();
        console.log('Doctor updated successfully:', data);
        alert('Doctor updated successfully!');
      } else {
        // Create new doctor
        const response = await fetch(`${BACKEND_BASE}/api/doctors`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(doctorPayload),
        });

        console.log('Create response status:', response.status);

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ message: 'Failed to create doctor' }));
          console.error('Create error:', errorData);
          throw new Error(errorData.message || 'Failed to create doctor');
        }

        const data = await response.json();
        console.log('Doctor created successfully:', data);
        alert('Doctor added successfully!');
      }

      onSuccess();
    } catch (error) {
      console.error('Error saving doctor:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to save doctor information. Please try again.';
      setErrors({ submit: errorMessage });
      alert(`Error: ${errorMessage}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-16">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">
              {isEditing ? 'Edit Doctor Profile' : 'Add New Doctor'}
            </h1>
            <p className="text-slate-600">
              {isEditing ? 'Update doctor information' : 'Fill in the details to add a new doctor'}
            </p>
          </div>
          <Button onClick={onCancel} variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50">
            Cancel
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Image Upload */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-900">
                <User className="w-5 h-5 text-blue-600" />
                Profile Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-6">
                <Avatar className="w-24 h-24 border-4 border-blue-50 ring-2 ring-blue-100">
                  <AvatarImage src={formData.imagePreview} />
                  <AvatarFallback className="bg-blue-100 text-blue-700">
                    <User className="w-8 h-8" />
                  </AvatarFallback>
                </Avatar>
                <div className="flex-1">
                  <Label htmlFor="image-upload" className="cursor-pointer">
                    <div className="border-2 border-dashed border-blue-200 rounded-lg p-6 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-colors bg-white/50">
                      <Upload className="w-8 h-8 mx-auto text-blue-400 mb-2" />
                      <p className="text-sm text-slate-600">
                        Click to upload image or drag and drop
                      </p>
                      <p className="text-xs text-slate-500 mt-1">
                        JPEG, PNG, WebP up to 5MB
                      </p>
                    </div>
                  </Label>
                  <input
                    id="image-upload"
                    type="file"
                    accept="image/jpeg,image/jpg,image/png,image/webp"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                  {errors.image && (
                    <p className="text-red-600 text-sm mt-2">{errors.image}</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Basic Information */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="text-slate-900">Basic Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="firstName" className="text-slate-700">First Name *</Label>
                  <Input
                    id="firstName"
                    value={formData.firstName}
                    onChange={(e) => setFormData(prev => ({ ...prev, firstName: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.firstName ? 'border-red-500' : ''}`}
                  />
                  {errors.firstName && (
                    <p className="text-red-600 text-sm mt-1">{errors.firstName}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="lastName" className="text-slate-700">Last Name *</Label>
                  <Input
                    id="lastName"
                    value={formData.lastName}
                    onChange={(e) => setFormData(prev => ({ ...prev, lastName: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.lastName ? 'border-red-500' : ''}`}
                  />
                  {errors.lastName && (
                    <p className="text-red-600 text-sm mt-1">{errors.lastName}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="gender" className="text-slate-700">Gender *</Label>
                  <Select value={formData.gender} onValueChange={(value) => setFormData(prev => ({ ...prev, gender: value as 'Male' | 'Female' | 'Other' }))}>
                    <SelectTrigger className={`border-blue-100 focus:border-blue-400 ${errors.gender ? 'border-red-500' : ''}`}>
                      <SelectValue placeholder="Select gender" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                  {errors.gender && (
                    <p className="text-red-600 text-sm mt-1">{errors.gender}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="age" className="text-slate-700">Age *</Label>
                  <Input
                    id="age"
                    type="number"
                    min="25"
                    max="80"
                    value={formData.age}
                    onChange={(e) => setFormData(prev => ({ ...prev, age: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.age ? 'border-red-500' : ''}`}
                  />
                  {errors.age && (
                    <p className="text-red-600 text-sm mt-1">{errors.age}</p>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Professional Information */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-900">
                <Building className="w-5 h-5 text-blue-600" />
                Professional Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="hospital" className="text-slate-700">Hospital *</Label>
                  <Input
                    id="hospital"
                    value={formData.hospital}
                    onChange={(e) => setFormData(prev => ({ ...prev, hospital: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.hospital ? 'border-red-500' : ''}`}
                  />
                  {errors.hospital && (
                    <p className="text-red-600 text-sm mt-1">{errors.hospital}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="position" className="text-slate-700">Position *</Label>
                  <Input
                    id="position"
                    value={formData.position}
                    onChange={(e) => setFormData(prev => ({ ...prev, position: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.position ? 'border-red-500' : ''}`}
                  />
                  {errors.position && (
                    <p className="text-red-600 text-sm mt-1">{errors.position}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="experience" className="text-slate-700">Years of Experience *</Label>
                  <Input
                    id="experience"
                    type="number"
                    min="0"
                    value={formData.experience}
                    onChange={(e) => setFormData(prev => ({ ...prev, experience: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.experience ? 'border-red-500' : ''}`}
                  />
                  {errors.experience && (
                    <p className="text-red-600 text-sm mt-1">{errors.experience}</p>
                  )}
                </div>

                <div>
                  <Label htmlFor="consultationFee" className="text-slate-700">Consultation Fee ($) *</Label>
                  <Input
                    id="consultationFee"
                    type="number"
                    min="0"
                    value={formData.consultationFee}
                    onChange={(e) => setFormData(prev => ({ ...prev, consultationFee: e.target.value }))}
                    className={`border-blue-100 focus:border-blue-400 ${errors.consultationFee ? 'border-red-500' : ''}`}
                  />
                  {errors.consultationFee && (
                    <p className="text-red-600 text-sm mt-1">{errors.consultationFee}</p>
                  )}
                </div>
              </div>

              <div>
                <Label htmlFor="location" className="text-slate-700">Location *</Label>
                <Input
                  id="location"
                  value={formData.location}
                  onChange={(e) => setFormData(prev => ({ ...prev, location: e.target.value }))}
                  className={`border-blue-100 focus:border-blue-400 ${errors.location ? 'border-red-500' : ''}`}
                  placeholder="City, Country"
                />
                {errors.location && (
                  <p className="text-red-600 text-sm mt-1">{errors.location}</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Education & Degrees */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-900">
                <GraduationCap className="w-5 h-5 text-blue-600" />
                Education & Degrees
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  value={newDegree}
                  onChange={(e) => setNewDegree(e.target.value)}
                  placeholder="Add degree (e.g., MD, PhD in Cardiology)"
                  className="border-blue-100 focus:border-blue-400"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      addToArray('degrees', newDegree);
                      setNewDegree('');
                    }
                  }}
                />
                <Button
                  type="button"
                  onClick={() => {
                    addToArray('degrees', newDegree);
                    setNewDegree('');
                  }}
                  disabled={!newDegree.trim()}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Plus className="w-4 h-4" />
                </Button>
              </div>

              <div className="flex flex-wrap gap-2">
                {formData.degrees.map((degree, index) => (
                  <Badge key={index} variant="secondary" className="flex items-center gap-1 bg-blue-50 text-blue-700 border border-blue-100">
                    {degree}
                    <button
                      type="button"
                      onClick={() => removeFromArray('degrees', index)}
                      className="ml-1 hover:text-red-600"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </Badge>
                ))}
              </div>
              {errors.degrees && (
                <p className="text-red-600 text-sm">{errors.degrees}</p>
              )}
            </CardContent>
          </Card>

          {/* Contact Information */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-900">
                <Mail className="w-5 h-5 text-blue-600" />
                Contact Information
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Emails */}
              <div>
                <Label className="text-slate-700">Email Addresses *</Label>
                <div className="space-y-2">
                  {formData.emails.map((email, index) => (
                    <div key={index} className="flex gap-2">
                      <Input
                        type="email"
                        value={email}
                        onChange={(e) => updateEmail(index, e.target.value)}
                        placeholder="doctor@hospital.com"
                        className={`border-blue-100 focus:border-blue-400 ${errors.emails ? 'border-red-500' : ''}`}
                      />
                      {formData.emails.length > 1 && (
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          onClick={() => {
                            const newEmails = formData.emails.filter((_, i) => i !== index);
                            setFormData(prev => ({ ...prev, emails: newEmails }));
                          }}
                          className="border-blue-200 text-blue-700 hover:bg-blue-50"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                  <Button type="button" variant="outline" onClick={addEmail} className="text-sm border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Plus className="w-4 h-4 mr-1" />
                    Add Another Email
                  </Button>
                </div>
                {errors.emails && (
                  <p className="text-red-600 text-sm mt-1">{errors.emails}</p>
                )}
              </div>

              {/* Phones */}
              <div>
                <Label className="text-slate-700">Phone Numbers *</Label>
                <div className="space-y-2">
                  {formData.phones.map((phone, index) => (
                    <div key={index} className="flex gap-2">
                      <Input
                        type="tel"
                        value={phone}
                        onChange={(e) => updatePhone(index, e.target.value)}
                        placeholder="+1 (555) 123-4567"
                        className={`border-blue-100 focus:border-blue-400 ${errors.phones ? 'border-red-500' : ''}`}
                      />
                      {formData.phones.length > 1 && (
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          onClick={() => {
                            const newPhones = formData.phones.filter((_, i) => i !== index);
                            setFormData(prev => ({ ...prev, phones: newPhones }));
                          }}
                          className="border-blue-200 text-blue-700 hover:bg-blue-50"
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                  <Button type="button" variant="outline" onClick={addPhone} className="text-sm border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Plus className="w-4 h-4 mr-1" />
                    Add Another Phone
                  </Button>
                </div>
                {errors.phones && (
                  <p className="text-red-600 text-sm mt-1">{errors.phones}</p>
                )}
              </div>

              {/* Website & LinkedIn */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="website" className="text-slate-700">Website</Label>
                  <Input
                    id="website"
                    type="url"
                    value={formData.website}
                    onChange={(e) => setFormData(prev => ({ ...prev, website: e.target.value }))}
                    placeholder="https://doctor-website.com"
                    className="border-blue-100 focus:border-blue-400"
                  />
                </div>

                <div>
                  <Label htmlFor="linkedIn" className="text-slate-700">LinkedIn Profile</Label>
                  <Input
                    id="linkedIn"
                    type="url"
                    value={formData.linkedIn}
                    onChange={(e) => setFormData(prev => ({ ...prev, linkedIn: e.target.value }))}
                    placeholder="https://linkedin.com/in/doctor"
                    className="border-blue-100 focus:border-blue-400"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Specialties */}
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-slate-900">
                <Award className="w-5 h-5 text-blue-600" />
                Fields of Expertise
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label className="text-slate-700">Add Specialty</Label>
                <Select value={newSpecialty} onValueChange={setNewSpecialty}>
                  <SelectTrigger className="border-blue-100 focus:border-blue-400">
                    <SelectValue placeholder="Select a specialty" />
                  </SelectTrigger>
                  <SelectContent>
                    {specialties
                      .filter(s => !formData.specialties.includes(s))
                      .map((specialty) => (
                        <SelectItem key={specialty} value={specialty}>
                          {specialty}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  onClick={() => {
                    if (newSpecialty) {
                      addToArray('specialties', newSpecialty);
                      setNewSpecialty('');
                    }
                  }}
                  disabled={!newSpecialty}
                  className="mt-2 bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Plus className="w-4 h-4 mr-1" />
                  Add Specialty
                </Button>
              </div>

              <div className="flex flex-wrap gap-2">
                {formData.specialties.map((specialty, index) => (
                  <Badge key={index} variant="secondary" className="flex items-center gap-1 bg-teal-50 text-teal-700 border border-teal-100">
                    {specialty}
                    <button
                      type="button"
                      onClick={() => removeFromArray('specialties', index)}
                      className="ml-1 hover:text-red-600"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </Badge>
                ))}
              </div>
              {errors.specialties && (
                <p className="text-red-600 text-sm">{errors.specialties}</p>
              )}
            </CardContent>
          </Card>

          {/* Languages & Awards */}
          <Card>
            <CardHeader>
              <CardTitle>Additional Information</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Languages */}
              <div>
                <Label>Languages *</Label>
                <div className="flex gap-2">
                  <Input
                    value={newLanguage}
                    onChange={(e) => setNewLanguage(e.target.value)}
                    placeholder="Add language (e.g., English, Spanish)"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        addToArray('languages', newLanguage);
                        setNewLanguage('');
                      }
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => {
                      addToArray('languages', newLanguage);
                      setNewLanguage('');
                    }}
                    disabled={!newLanguage.trim()}
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>

                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.languages.map((language, index) => (
                    <Badge key={index} variant="outline" className="flex items-center gap-1">
                      {language}
                      <button
                        type="button"
                        onClick={() => removeFromArray('languages', index)}
                        className="ml-1 hover:text-red-600"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
                {errors.languages && (
                  <p className="text-red-600 text-sm mt-1">{errors.languages}</p>
                )}
              </div>

              {/* Awards */}
              <div>
                <Label>Awards & Recognition</Label>
                <div className="flex gap-2">
                  <Input
                    value={newAward}
                    onChange={(e) => setNewAward(e.target.value)}
                    placeholder="Add award or recognition"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        addToArray('awards', newAward);
                        setNewAward('');
                      }
                    }}
                  />
                  <Button
                    type="button"
                    onClick={() => {
                      addToArray('awards', newAward);
                      setNewAward('');
                    }}
                    disabled={!newAward.trim()}
                  >
                    <Plus className="w-4 h-4" />
                  </Button>
                </div>

                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.awards.map((award, index) => (
                    <Badge key={index} variant="outline" className="flex items-center gap-1">
                      {award}
                      <button
                        type="button"
                        onClick={() => removeFromArray('awards', index)}
                        className="ml-1 hover:text-red-600"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>

              {/* Biography */}
              <div>
                <Label htmlFor="biography">Biography</Label>
                <Textarea
                  id="biography"
                  value={formData.biography}
                  onChange={(e) => setFormData(prev => ({ ...prev, biography: e.target.value }))}
                  placeholder="Brief professional biography and background..."
                  rows={4}
                />
              </div>
            </CardContent>
          </Card>

          {/* Error Messages */}
          {errors.submit && (
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{errors.submit}</AlertDescription>
            </Alert>
          )}

          {/* Submit Buttons */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex gap-4 justify-end">
                <Button type="button" variant="outline" onClick={onCancel}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isSubmitting} className="flex items-center gap-2">
                  {isSubmitting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      {isEditing ? 'Update Doctor' : 'Add Doctor'}
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>
        </form>
      </div>
    </div>
  );
}