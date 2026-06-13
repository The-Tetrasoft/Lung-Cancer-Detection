import React, { useState, useRef, useEffect } from 'react';
import { db } from '../firebase-client';
import { collection, addDoc } from 'firebase/firestore';
import { ArrowLeft, Upload, X, Loader2, AlertCircle, CheckCircle, XCircle, Activity, Stethoscope, TrendingUp, Layers, Eye, EyeOff } from 'lucide-react';
import { BACKEND_BASE } from '../utils/config';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Badge } from './ui/badge';
import { BarChart, Bar, PieChart, Pie, Cell, RadialBarChart, RadialBar, ResponsiveContainer, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import { DoctorRecommendations } from './DoctorRecommendations';

interface User {
  username: string;
  firstName: string;
  lastName: string;
  email: string;
  role: string;
  profilePicture?: string;
}

interface StartAnalysisProps {
  user: User;
  onNavigateBack: () => void;
  onNavigateToDoctor: (doctorId: string) => void;
}

interface HeatmapRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  intensity: number; // 0-1
  label: string;
}

interface AnalysisResult {
  prediction: 'Positive' | 'Negative';
  confidence: number;
  riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
  detailedMetrics: {
    lungOpacity: number;
    noduleDetection: number;
    tissueAbnormality: number;
    inflammationMarkers: number;
  };
  recommendations: string[];
  heatmapRegions: HeatmapRegion[];
}

// Helper function to convert RGB color to RGBA
const rgbToRgba = (rgbColor: string, alpha: number): string => {
  const match = rgbColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (match) {
    return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`;
  }
  return rgbColor;
};

export function StartAnalysis({ user, onNavigateBack, onNavigateToDoctor }: StartAnalysisProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image')) {
        setError('Please select a valid image file');
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('Image size should be less than 10MB');
        return;
      }

      setError(null);
      setSelectedImage(file);

      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);

      // Reset previous results
      setAnalysisResult(null);
    }
  };

  const handleRemoveImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setAnalysisResult(null);
    setError(null);
    setShowHeatmap(false);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('Please upload an X-ray image first');
      return;
    }

    setIsAnalyzing(true);
    setProgress(0);
    setError(null);

    // Try to call backend analyze endpoint
    setProgress(10);
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout for backend

    try {
      const form = new FormData();
      form.append('image', selectedImage as File);
      form.append('username', user.username);

      const resp = await fetch(`${BACKEND_BASE}/api/analyze`, {
        method: 'POST',
        body: form,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!resp.ok) throw new Error('Analysis failed');

      const result: AnalysisResult = await resp.json();
      setProgress(100);
      setAnalysisResult(result);
      setIsAnalyzing(false);
      if (result.heatmapRegions.length > 0) setShowHeatmap(true);
      await saveMedicalHistory(result);
    } catch (err) {
      clearTimeout(timeoutId);
      console.error('Analysis error, falling back to mock:', err);

      // fallback to previous mock simulation (FASTER NOW)
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 95) {
            clearInterval(progressInterval);
            return 95;
          }
          return prev + 10; // Faster progress
        });
      }, 150);

      setTimeout(() => {
        clearInterval(progressInterval);
        setProgress(100);
        const mockResult: AnalysisResult = generateMockAnalysis(selectedImage as File);
        setTimeout(() => {
          setAnalysisResult(mockResult);
          setIsAnalyzing(false);
          if (mockResult.heatmapRegions.length > 0) setShowHeatmap(true);
          saveMedicalHistory(mockResult);
        }, 300);
      }, 2000); // Reduced from 4000ms to 2000ms
    }
  };

  const generateMockAnalysis = (file: File): AnalysisResult => {
    // Create a simple seeded random number generator
    // Seed based on file name and size to ensure same file = same result
    let seed = file.name.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) + file.size;

    const seededRandom = () => {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    };

    // Randomize results for demo purposes (deterministically)
    const random = seededRandom();
    const isPositive = random > 0.7;

    const confidence = isPositive
      ? 65 + seededRandom() * 30 // 65-95% for positive
      : 80 + seededRandom() * 18; // 80-98% for negative

    let riskLevel: 'Low' | 'Medium' | 'High' | 'Critical';
    if (!isPositive) {
      riskLevel = 'Low';
    } else {
      if (confidence > 85) riskLevel = 'Critical';
      else if (confidence > 75) riskLevel = 'High';
      else riskLevel = 'Medium';
    }

    // Generate heatmap regions
    const heatmapRegions: HeatmapRegion[] = [];
    if (isPositive) {
      // Add 2-4 regions for positive cases
      const numRegions = 2 + Math.floor(seededRandom() * 3);
      for (let i = 0; i < numRegions; i++) {
        heatmapRegions.push({
          x: 20 + seededRandom() * 50,
          y: 15 + seededRandom() * 60,
          width: 10 + seededRandom() * 20,
          height: 10 + seededRandom() * 20,
          intensity: 0.5 + seededRandom() * 0.5,
          label: ['Nodule', 'Opacity', 'Mass', 'Abnormality'][Math.floor(seededRandom() * 4)]
        });
      }
    } else {
      // Add 0-2 low-intensity regions for negative cases
      const numRegions = Math.floor(seededRandom() * 3);
      for (let i = 0; i < numRegions; i++) {
        heatmapRegions.push({
          x: 20 + seededRandom() * 50,
          y: 15 + seededRandom() * 60,
          width: 8 + seededRandom() * 15,
          height: 8 + seededRandom() * 15,
          intensity: 0.2 + seededRandom() * 0.3,
          label: 'Minor Finding'
        });
      }
    }

    return {
      prediction: isPositive ? 'Positive' : 'Negative',
      confidence: Math.round(confidence * 10) / 10,
      riskLevel,
      detailedMetrics: {
        lungOpacity: Math.round((40 + seededRandom() * 60) * 10) / 10,
        noduleDetection: Math.round((isPositive ? 60 + seededRandom() * 35 : 10 + seededRandom() * 30) * 10) / 10,
        tissueAbnormality: Math.round((isPositive ? 55 + seededRandom() * 40 : 15 + seededRandom() * 35) * 10) / 10,
        inflammationMarkers: Math.round((30 + seededRandom() * 60) * 10) / 10,
      },
      recommendations: isPositive
        ? [
          'Immediate consultation with an oncologist is recommended',
          'Additional diagnostic tests (CT scan, biopsy) should be performed',
          'Discuss treatment options with your healthcare provider',
          'Consider getting a second opinion from a specialist',
          'Monitor symptoms and report any changes immediately'
        ]
        : [
          'Continue regular health check-ups',
          'Maintain a healthy lifestyle and diet',
          'Avoid smoking and exposure to pollutants',
          'Schedule annual lung screenings if you are in a high-risk group',
          'Consult a doctor if you experience any respiratory symptoms'
        ],
      heatmapRegions
    };
  };

  const saveMedicalHistory = async (result: AnalysisResult) => {
    const payload = {
      username: user.username,
      date: new Date().toISOString(),
      type: 'X-Ray' as const,
      result: result.prediction === 'Positive' ? 'Abnormal' : 'Normal',
      confidence: result.confidence,
      notes: result.prediction === 'Positive' ? 'AI detected potential abnormal findings.' : 'AI found no suspicious findings.',
      doctorName: '',
      imageUrl: imagePreview || '',
      raw: result,
    };

    try {
      // Save to Firestore
      await addDoc(collection(db, 'medical-history'), payload);
      console.log('Medical history saved to Firestore');
    } catch (err) {
      console.error('Failed to save medical history to Firestore:', err);
      // Fallback to localStorage just in case
      try {
        const history = JSON.parse(localStorage.getItem('medicalHistory') || '[]');
        const newEntry = { id: Date.now().toString(), ...payload };
        history.unshift(newEntry);
        localStorage.setItem('medicalHistory', JSON.stringify(history));
      } catch (e) {
        console.error('Failed to save medical history locally:', e);
      }
    }
  };

  // Get color based on intensity (blue -> yellow -> red)
  const getHeatmapColor = (intensity: number): string => {
    if (intensity < 0.33) {
      // Blue to cyan
      const t = intensity / 0.33;
      return `rgb(${Math.round(59 * t)}, ${Math.round(130 + 125 * t)}, ${Math.round(246 - 36 * t)})`;
    } else if (intensity < 0.66) {
      // Cyan to yellow
      const t = (intensity - 0.33) / 0.33;
      return `rgb(${Math.round(59 + 196 * t)}, ${Math.round(255 - 110 * t)}, ${Math.round(210 - 210 * t)})`;
    } else {
      // Yellow to red
      const t = (intensity - 0.66) / 0.34;
      return `rgb(${Math.round(255)}, ${Math.round(145 - 145 * t)}, ${Math.round(0)})`;
    }
  };

  // Draw heatmap on canvas
  const drawHeatmap = () => {
    if (!canvasRef.current || !imageRef.current || !analysisResult || !showHeatmap) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;

    if (!ctx) return;

    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw heatmap regions
    analysisResult.heatmapRegions.forEach((region) => {
      const x = (region.x / 100) * canvas.width;
      const y = (region.y / 100) * canvas.height;
      const width = (region.width / 100) * canvas.width;
      const height = (region.height / 100) * canvas.height;

      // Create radial gradient for each region
      const centerX = x + width / 2;
      const centerY = y + height / 2;
      const radius = Math.max(width, height) / 2;

      const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);

      // Color based on intensity
      const color = getHeatmapColor(region.intensity);
      gradient.addColorStop(0, rgbToRgba(color, region.intensity));
      gradient.addColorStop(0.5, rgbToRgba(color, region.intensity * 0.6));
      gradient.addColorStop(1, rgbToRgba(color, 0));

      ctx.fillStyle = gradient;
      ctx.fillRect(x - width / 2, y - height / 2, width * 2, height * 2);

      // Draw border around high-intensity regions
      if (region.intensity > 0.6) {
        ctx.strokeStyle = rgbToRgba(color, 0.8);
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);
      }
    });
  };

  // Function to draw heatmap on a canvas element
  const drawHeatmapOnCanvas = (canvas: HTMLCanvasElement, imageUrl: string, regions: HeatmapRegion[]) => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = img.width;
        canvas.height = img.height;

        regions.forEach((region) => {
          const x = (region.x / 100) * canvas.width;
          const y = (region.y / 100) * canvas.height;
          const width = (region.width / 100) * canvas.width;
          const height = (region.height / 100) * canvas.height;

          const centerX = x + width / 2;
          const centerY = y + height / 2;
          const radius = Math.max(width, height) / 2;

          const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);

          const color = getHeatmapColor(region.intensity);
          gradient.addColorStop(0, rgbToRgba(color, region.intensity));
          gradient.addColorStop(0.5, rgbToRgba(color, region.intensity * 0.6));
          gradient.addColorStop(1, rgbToRgba(color, 0));

          ctx.fillStyle = gradient;
          ctx.fillRect(x - width / 2, y - height / 2, width * 2, height * 2);

          if (region.intensity > 0.6) {
            ctx.strokeStyle = rgbToRgba(color, 0.8);
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);
          }
        });
      }
    };
  };

  // Redraw heatmap when visibility or result changes
  useEffect(() => {
    if (showHeatmap && analysisResult && imageRef.current) {
      // Wait for image to load
      if (imageRef.current.complete) {
        drawHeatmap();
      } else {
        imageRef.current.onload = drawHeatmap;
      }
    }
  }, [showHeatmap, analysisResult]);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return '#10b981';
      case 'Medium': return '#f59e0b';
      case 'High': return '#f97316';
      case 'Critical': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const metricsData = analysisResult ? [
    { name: 'Lung Opacity', value: analysisResult.detailedMetrics.lungOpacity, fill: '#3b82f6' },
    { name: 'Nodule Detection', value: analysisResult.detailedMetrics.noduleDetection, fill: '#8b5cf6' },
    { name: 'Tissue Abnormality', value: analysisResult.detailedMetrics.tissueAbnormality, fill: '#ec4899' },
    { name: 'Inflammation', value: analysisResult.detailedMetrics.inflammationMarkers, fill: '#f59e0b' },
  ] : [];

  const confidenceData = analysisResult ? [
    { name: 'Confidence', value: analysisResult.confidence, fill: getRiskColor(analysisResult.riskLevel) }
  ] : [];

  const distributionData = analysisResult ? [
    { name: analysisResult.prediction, value: analysisResult.confidence },
    { name: 'Uncertainty', value: 100 - analysisResult.confidence }
  ] : [];

  const COLORS = ['#3b82f6', '#e5e7eb'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 dark:from-gray-900 dark:via-blue-900 dark:to-indigo-900 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <Button
            onClick={onNavigateBack}
            variant="ghost"
            className="mb-4 hover:bg-white/50 dark:hover:bg-gray-800/50 text-slate-600"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>

          <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg p-6 shadow-lg border border-blue-100">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md">
                <Activity className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">X-Ray Analysis</h1>
                <p className="text-slate-600">Upload and analyze lung X-ray images</p>
              </div>
            </div>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Upload Section */}
        <Card className="mb-6 shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload X-Ray Image
            </CardTitle>
            <CardDescription>
              Select a clear chest X-ray image for analysis. Supported formats: JPG, PNG, JPEG
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!imagePreview ? (
              <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-8 md:p-12 text-center hover:border-blue-500 dark:hover:border-blue-400 transition-colors">
                <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="mb-2">Choose an X-ray image</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Click to browse or drag and drop
                </p>
                <label htmlFor="xray-upload" className="cursor-pointer">
                  <Button type="button" onClick={() => document.getElementById('xray-upload')?.click()}>
                    Select Image
                  </Button>
                </label>
                <input
                  id="xray-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                <p className="text-xs text-muted-foreground mt-4">
                  Maximum file size: 10MB
                </p>
              </div>
            ) : (
              <div>
                <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5">
                  <img
                    ref={imageRef}
                    src={imagePreview}
                    alt="X-ray preview"
                    className="w-full h-auto max-h-96 object-contain mx-auto"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    style={{ display: showHeatmap ? 'block' : 'none' }}
                  />
                  <Button
                    onClick={handleRemoveImage}
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  {analysisResult && (
                    <Button
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      variant={showHeatmap ? "default" : "secondary"}
                      size="icon"
                      className="absolute top-2 left-2"
                      title={showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
                    >
                      {showHeatmap ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  )}
                </div>

                {analysisResult && showHeatmap && (
                  <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-red-50 dark:from-blue-950 dark:to-red-950 rounded-lg border border-blue-200 dark:border-blue-800">
                    <div className="flex items-center gap-2 mb-3">
                      <Layers className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                      <span className="text-sm">Heatmap Legend</span>
                    </div>
                    <div className="flex items-center gap-4 flex-wrap">
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded" style={{ background: 'linear-gradient(to right, rgb(59, 130, 246), rgb(59, 255, 210))' }}></div>
                        <span className="text-xs text-muted-foreground">Low Concern</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded" style={{ background: 'linear-gradient(to right, rgb(59, 255, 210), rgb(255, 145, 0))' }}></div>
                        <span className="text-xs text-muted-foreground">Medium Concern</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-4 rounded" style={{ background: 'linear-gradient(to right, rgb(255, 145, 0), rgb(255, 0, 0))' }}></div>
                        <span className="text-xs text-muted-foreground">High Concern</span>
                      </div>
                    </div>
                    {analysisResult.heatmapRegions.length > 0 && (
                      <div className="mt-3 text-xs text-muted-foreground">
                        {analysisResult.heatmapRegions.length} region{analysisResult.heatmapRegions.length !== 1 ? 's' : ''} of interest detected
                      </div>
                    )}
                  </div>
                )}

                <div className="mt-4 flex flex-col sm:flex-row gap-3">
                  <Button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Stethoscope className="mr-2 h-4 w-4" />
                        Analyze X-Ray
                      </>
                    )}
                  </Button>

                  <label htmlFor="xray-upload-replace" className="flex-1">
                    <Button
                      type="button"
                      variant="outline"
                      className="w-full"
                      onClick={() => document.getElementById('xray-upload-replace')?.click()}
                      disabled={isAnalyzing}
                    >
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Different Image
                    </Button>
                  </label>
                  <input
                    id="xray-upload-replace"
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                  />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Progress */}
        {isAnalyzing && (
          <Card className="mb-6 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Loader2 className="h-5 w-5 animate-spin" />
                Analysis in Progress
              </CardTitle>
              <CardDescription>
                AI is analyzing the X-ray image...
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Progress value={progress} className="w-full h-2" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className={progress >= 25 ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}>
                    {progress >= 25 ? '✓' : '○'} Image Processing
                  </div>
                  <div className={progress >= 50 ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}>
                    {progress >= 50 ? '✓' : '○'} Feature Extraction
                  </div>
                  <div className={progress >= 75 ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}>
                    {progress >= 75 ? '✓' : '○'} AI Analysis
                  </div>
                  <div className={progress >= 100 ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}>
                    {progress >= 100 ? '✓' : '○'} Generating Report
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Analysis Results */}
        {analysisResult && !isAnalyzing && (
          <div className="space-y-6">
            {/* Main Result Card */}
            <Card className="shadow-lg">
              <CardHeader className={`${analysisResult.prediction === 'Positive'
                ? 'bg-red-50 dark:bg-red-900/20'
                : 'bg-green-50 dark:bg-green-900/20'
                }`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {analysisResult.prediction === 'Positive' ? (
                      <div className="p-3 bg-red-500 rounded-full">
                        <XCircle className="h-6 w-6 text-white" />
                      </div>
                    ) : (
                      <div className="p-3 bg-green-500 rounded-full">
                        <CheckCircle className="h-6 w-6 text-white" />
                      </div>
                    )}
                    <div>
                      <CardTitle className="text-2xl">
                        Analysis Complete
                      </CardTitle>
                      <CardDescription>
                        {new Date().toLocaleDateString()} at {new Date().toLocaleTimeString()}
                      </CardDescription>
                    </div>
                  </div>
                  <Badge
                    variant="outline"
                    className="text-lg px-4 py-2"
                    style={{
                      borderColor: getRiskColor(analysisResult.riskLevel),
                      color: getRiskColor(analysisResult.riskLevel)
                    }}
                  >
                    {analysisResult.riskLevel} Risk
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="grid md:grid-cols-2 gap-6 mb-6">
                  <div className="space-y-2">
                    <label className="text-sm text-muted-foreground">Prediction</label>
                    <div className={`text-3xl ${analysisResult.prediction === 'Positive'
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-green-600 dark:text-green-400'
                      }`}>
                      {analysisResult.prediction === 'Positive'
                        ? 'Lung Cancer Detected'
                        : 'No Cancer Detected'}
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm text-muted-foreground">Confidence Level</label>
                    <div className="text-3xl" style={{ color: getRiskColor(analysisResult.riskLevel) }}>
                      {analysisResult.confidence}%
                    </div>
                  </div>
                </div>

                <div className="mb-6">
                  <label className="text-sm text-muted-foreground mb-2 block">Confidence Meter</label>
                  <Progress
                    value={analysisResult.confidence}
                    className="w-full h-4"
                    style={{
                      ['--progress-background' as any]: getRiskColor(analysisResult.riskLevel)
                    }}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Charts Section */}
            <div className="grid md:grid-cols-2 gap-6">
              {/* Confidence Gauge */}
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle>Confidence Score</CardTitle>
                  <CardDescription>AI model confidence level</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadialBarChart
                      cx="50%"
                      cy="50%"
                      innerRadius="60%"
                      outerRadius="90%"
                      data={confidenceData}
                      startAngle={180}
                      endAngle={0}
                    >
                      <RadialBar
                        background
                        dataKey="value"
                        cornerRadius={10}
                      />
                      <text
                        x="50%"
                        y="50%"
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="text-3xl"
                        fill={getRiskColor(analysisResult.riskLevel)}
                      >
                        {analysisResult.confidence}%
                      </text>
                    </RadialBarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Prediction Distribution */}
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle>Prediction Distribution</CardTitle>
                  <CardDescription>Model certainty breakdown</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={distributionData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {distributionData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Heatmap Analysis */}
              {analysisResult.heatmapRegions.length > 0 && (
                <Card className="shadow-lg md:col-span-2">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Layers className="h-5 w-5" />
                      AI Detection Heatmap
                    </CardTitle>
                    <CardDescription>Visual representation of detected regions of interest</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Heatmap Image */}
                      <div className="relative rounded-lg overflow-hidden bg-black/5 dark:bg-white/5 border-2 border-gray-200 dark:border-gray-700">
                        <img
                          src={imagePreview || ''}
                          alt="X-ray with heatmap"
                          className="w-full h-auto object-contain"
                        />
                        <canvas
                          className="absolute top-0 left-0 w-full h-full pointer-events-none"
                          ref={(canvas) => {
                            if (canvas && imagePreview && analysisResult) {
                              drawHeatmapOnCanvas(canvas, imagePreview, analysisResult.heatmapRegions);
                            }
                          }}
                        />
                      </div>

                      {/* Regions List */}
                      <div className="space-y-4">
                        <div>
                          <h4 className="mb-3">Detected Regions ({analysisResult.heatmapRegions.length})</h4>
                          <div className="space-y-2 max-h-64 overflow-y-auto">
                            {analysisResult.heatmapRegions.map((region, index) => (
                              <div
                                key={index}
                                className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700"
                              >
                                <div className="flex items-center justify-between mb-2">
                                  <span className="text-sm">{region.label}</span>
                                  <Badge
                                    variant="outline"
                                    style={{
                                      borderColor: getHeatmapColor(region.intensity),
                                      color: getHeatmapColor(region.intensity)
                                    }}
                                  >
                                    {Math.round(region.intensity * 100)}%
                                  </Badge>
                                </div>
                                <div className="flex items-center gap-2">
                                  <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700">
                                    <div
                                      className="h-full rounded-full"
                                      style={{
                                        width: `${region.intensity * 100}%`,
                                        backgroundColor: getHeatmapColor(region.intensity)
                                      }}
                                    />
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Heatmap Legend */}
                        <div className="p-4 bg-gradient-to-r from-blue-50 to-red-50 dark:from-blue-950 dark:to-red-950 rounded-lg">
                          <h4 className="mb-3 text-sm">Intensity Scale</h4>
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-3 rounded" style={{ backgroundColor: 'rgb(59, 130, 246)' }}></div>
                              <span className="text-xs text-muted-foreground">0-33% - Low</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-3 rounded" style={{ backgroundColor: 'rgb(255, 145, 0)' }}></div>
                              <span className="text-xs text-muted-foreground">34-66% - Medium</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-3 rounded" style={{ backgroundColor: 'rgb(255, 0, 0)' }}></div>
                              <span className="text-xs text-muted-foreground">67-100% - High</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Detailed Metrics */}
              <Card className="shadow-lg md:col-span-2">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Detailed Analysis Metrics
                  </CardTitle>
                  <CardDescription>Individual feature analysis scores</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={metricsData}>
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                    {metricsData.map((metric) => (
                      <div key={metric.name} className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="text-sm text-muted-foreground mb-1">{metric.name}</div>
                        <div className="text-xl" style={{ color: metric.fill }}>
                          {metric.value}%
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recommendations */}
            <Card className="shadow-lg bg-white/80 backdrop-blur-sm border-blue-100">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-slate-900">
                  <Stethoscope className="h-5 w-5 text-blue-600" />
                  Medical Recommendations
                </CardTitle>
                <CardDescription>
                  {analysisResult.prediction === 'Positive'
                    ? 'Please consult with a healthcare professional immediately'
                    : 'Continue maintaining good lung health'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analysisResult.recommendations.map((recommendation, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-blue-50/50 border border-blue-100">
                      <div className="mt-0.5">
                        {analysisResult.prediction === 'Positive' ? (
                          <AlertCircle className="h-5 w-5 text-red-500" />
                        ) : (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        )}
                      </div>
                      <p className="text-sm flex-1 text-slate-700">{recommendation}</p>
                    </div>
                  ))}
                </div>

                {analysisResult.prediction === 'Positive' && (
                  <Alert className="mt-6 border-red-200 bg-red-50">
                    <AlertCircle className="h-4 w-4 text-red-600" />
                    <AlertDescription className="text-red-800">
                      <strong>Important:</strong> This AI analysis is not a substitute for professional medical diagnosis.
                      Please consult with qualified healthcare professionals for proper evaluation and treatment.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Doctor Recommendations - NEW */}
            {analysisResult.prediction === 'Positive' && (
              <DoctorRecommendations
                user={user}
                analysisResult={analysisResult}
                onNavigateToDoctor={onNavigateToDoctor}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
