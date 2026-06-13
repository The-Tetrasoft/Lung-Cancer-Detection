// Mock data for doctors - this would typically come from a database

export interface Doctor {
  id: string;
  firstName: string;
  lastName: string;
  gender: 'Male' | 'Female' | 'Other';
  age: number;
  hospital: string;
  position: string;
  degrees: string[];
  location: string;
  email: string;
  phone: string;
  website?: string;
  linkedIn?: string;
  specialties: string[];
  imageUrl?: string;
  biography?: string;
  experience: number; // years
  languages: string[];
  awards?: string[];
  publications?: number;
  rating: number;
  consultationFee: number;
}

export const mockDoctors: Doctor[] = [
  {
    id: '1',
    firstName: 'Mohammad Abdul',
    lastName: 'Matin',
    gender: 'Male',
    age: 52,
    hospital: 'Newcastle University Hospital',
    position: 'Professor & Chair',
    degrees: ['PhD (Newcastle University, UK)', 'MD'],
    location: 'Newcastle, UK',
    email: 'mohammad.matin@northsouth.edu',
    phone: '+88 02 55668200 Ext – 1501, 1541',
    website: 'https://profile.example.com',
    linkedIn: 'https://linkedin.com/in/matin',
    specialties: ['RF Technology', 'Microwave and Communication Technology', 'Broadband Access', 'Communication Technologies'],
    imageUrl: 'doctor-1.jpg',
    biography: 'Dr. Mohammad Abdul Matin is a renowned professor specializing in RF and microwave technologies with over 20 years of experience in the field.',
    experience: 25,
    languages: ['English', 'Bengali', 'Arabic'],
    awards: ['Best Research Award 2020', 'Excellence in Teaching 2019'],
    publications: 150,
    rating: 4.8,
    consultationFee: 200
  },
  {
    id: '2',
    firstName: 'M.',
    lastName: 'Rokonuzzaman',
    gender: 'Male',
    age: 48,
    hospital: 'Memorial University of Newfoundland',
    position: 'Professor',
    degrees: ['Ph.D from Memorial University of Newfoundland, Canada'],
    location: 'Newfoundland, Canada',
    email: 'm.rokonuzzaman@northsouth.edu',
    phone: '+88 02 55668200 Ext – 1510',
    specialties: ['Artificial Intelligence & Robotics', 'Technology Transfer and Policy'],
    imageUrl: 'doctor-2.jpg',
    biography: 'Dr. Rokonuzzaman is a leading expert in AI and robotics with extensive research in technology policy and implementation.',
    experience: 20,
    languages: ['English', 'Bengali'],
    awards: ['Innovation Excellence Award 2021'],
    publications: 120,
    rating: 4.6,
    consultationFee: 180
  },
  {
    id: '3',
    firstName: 'K. M. A.',
    lastName: 'Salam',
    gender: 'Male',
    age: 45,
    hospital: 'Muroran Institute of Technology',
    position: 'Professor & Controller of Examinations',
    degrees: ['Ph.D from Muroran Institute of Tech., Japan'],
    location: 'Tokyo, Japan',
    email: 'kazi.salam@northsouth.edu',
    phone: '+88 02 55668200 Ext – 1509',
    specialties: ['Power Systems and Renewable Energy', 'Semiconductor Device and Technology'],
    imageUrl: 'doctor-3.jpg',
    biography: 'Dr. Salam specializes in renewable energy systems and semiconductor technology with international research collaborations.',
    experience: 18,
    languages: ['English', 'Japanese', 'Bengali'],
    awards: ['Renewable Energy Research Excellence 2020'],
    publications: 95,
    rating: 4.7,
    consultationFee: 220
  },
  {
    id: '4',
    firstName: 'Sarah',
    lastName: 'Johnson',
    gender: 'Female',
    age: 42,
    hospital: 'Johns Hopkins Hospital',
    position: 'Chief of Pulmonology',
    degrees: ['MD', 'PhD in Pulmonary Medicine'],
    location: 'Baltimore, MD, USA',
    email: 'sarah.johnson@jhh.edu',
    phone: '+1 410 955 5000',
    website: 'https://hopkinsmedicine.org/profiles/johnson',
    specialties: ['Lung Cancer', 'Pulmonary Medicine', 'Thoracic Oncology', 'Interventional Pulmonology'],
    imageUrl: 'doctor-4.jpg',
    biography: 'Dr. Johnson is a board-certified pulmonologist specializing in lung cancer diagnosis and treatment with over 15 years of clinical experience.',
    experience: 15,
    languages: ['English', 'Spanish'],
    awards: ['Outstanding Physician Award 2022', 'Best Clinical Research 2021'],
    publications: 85,
    rating: 4.9,
    consultationFee: 350
  },
  {
    id: '5',
    firstName: 'Michael',
    lastName: 'Chen',
    gender: 'Male',
    age: 38,
    hospital: 'Mayo Clinic',
    position: 'Senior Radiologist',
    degrees: ['MD', 'Fellowship in Thoracic Radiology'],
    location: 'Rochester, MN, USA',
    email: 'michael.chen@mayo.edu',
    phone: '+1 507 284 2511',
    specialties: ['Thoracic Imaging', 'Lung Cancer Screening', 'CT Imaging', 'MRI'],
    imageUrl: 'doctor-5.jpg',
    biography: 'Dr. Chen is an expert in thoracic imaging with specialized training in early lung diagnosis using advanced imaging techniques.',
    experience: 12,
    languages: ['English', 'Mandarin'],
    awards: ['Radiology Excellence Award 2023'],
    publications: 60,
    rating: 4.8,
    consultationFee: 300
  },
  {
    id: '6',
    firstName: 'Emily',
    lastName: 'Rodriguez',
    gender: 'Female',
    age: 39,
    hospital: 'MD Anderson Cancer Center',
    position: 'Associate Professor of Oncology',
    degrees: ['MD', 'PhD in Oncology'],
    location: 'Houston, TX, USA',
    email: 'emily.rodriguez@mdanderson.org',
    phone: '+1 713 792 2121',
    specialties: ['Lung Cancer', 'Medical Oncology', 'Immunotherapy', 'Clinical Trials'],
    imageUrl: 'doctor-6.jpg',
    biography: 'Dr. Rodriguez is a medical oncologist specializing in lung cancer treatment with focus on immunotherapy and personalized medicine.',
    experience: 14,
    languages: ['English', 'Spanish', 'Portuguese'],
    awards: ['Cancer Research Excellence 2022', 'Clinical Innovation Award 2020'],
    publications: 110,
    rating: 4.9,
    consultationFee: 400
  },
  {
    id: '7',
    firstName: 'David',
    lastName: 'Wilson',
    gender: 'Male',
    age: 55,
    hospital: 'Cleveland Clinic',
    position: 'Director of Thoracic Surgery',
    degrees: ['MD', 'Fellowship in Thoracic Surgery'],
    location: 'Cleveland, OH, USA',
    email: 'david.wilson@ccf.org',
    phone: '+1 216 444 2200',
    specialties: ['Thoracic Surgery', 'Lung Cancer Surgery', 'Minimally Invasive Surgery', 'Robotic Surgery'],
    biography: 'Dr. Wilson is a leading thoracic surgeon with expertise in minimally invasive and robotic lung cancer surgery.',
    experience: 22,
    languages: ['English'],
    awards: ['Surgical Excellence Award 2021', 'Innovation in Surgery 2019'],
    publications: 140,
    rating: 4.8,
    consultationFee: 450
  },
  {
    id: '8',
    firstName: 'Lisa',
    lastName: 'Park',
    gender: 'Female',
    age: 44,
    hospital: 'Stanford University Medical Center',
    position: 'Associate Professor of Pathology',
    degrees: ['MD', 'PhD in Pathology'],
    location: 'Stanford, CA, USA',
    email: 'lisa.park@stanford.edu',
    phone: '+1 650 723 4000',
    specialties: ['Lung Pathology', 'Cancer Diagnosis', 'Molecular Pathology', 'Biomarker Analysis'],
    biography: 'Dr. Park is a pathologist specializing in lung cancer diagnosis and molecular characterization for personalized treatment.',
    experience: 16,
    languages: ['English', 'Korean'],
    awards: ['Pathology Research Award 2023'],
    publications: 75,
    rating: 4.7,
    consultationFee: 280
  }
];

export const specialties = [
  'Lung Cancer',
  'Pulmonary Medicine',
  'Thoracic Oncology',
  'Medical Oncology',
  'Thoracic Surgery',
  'Radiology',
  'Pathology',
  'Interventional Pulmonology',
  'Thoracic Imaging',
  'Immunotherapy',
  'Clinical Trials',
  'Minimally Invasive Surgery',
  'Robotic Surgery',
  'CT Imaging',
  'MRI',
  'Lung Cancer Screening',
  'Molecular Pathology',
  'Biomarker Analysis',
  'RF Technology',
  'Artificial Intelligence & Robotics',
  'Power Systems and Renewable Energy'
];

export function getDoctorsBySpecialty(specialty: string): Doctor[] {
  return mockDoctors.filter(doctor => 
    doctor.specialties.some(s => 
      s.toLowerCase().includes(specialty.toLowerCase())
    )
  );
}

export function getDoctorById(id: string): Doctor | undefined {
  return mockDoctors.find(doctor => doctor.id === id);
}

export function searchDoctors(query: string): Doctor[] {
  const searchTerm = query.toLowerCase();
  return mockDoctors.filter(doctor => 
    doctor.firstName.toLowerCase().includes(searchTerm) ||
    doctor.lastName.toLowerCase().includes(searchTerm) ||
    doctor.hospital.toLowerCase().includes(searchTerm) ||
    doctor.specialties.some(s => s.toLowerCase().includes(searchTerm)) ||
    doctor.location.toLowerCase().includes(searchTerm)
  );
}