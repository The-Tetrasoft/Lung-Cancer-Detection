import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { FileText, Calendar, Download, Eye, Filter, Search } from 'lucide-react';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { BACKEND_BASE } from '../utils/config';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import jsPDF from 'jspdf';
import { Capacitor } from '@capacitor/core';
import { Filesystem, Directory, Encoding } from '@capacitor/filesystem';
import { db } from '../firebase-client';
import { collection, query, where, getDocs } from 'firebase/firestore';

// ... (existing imports)

// Helper to save file on both Web and Native
const saveFile = async (fileName: string, data: string, contentType: string, isBase64: boolean = false) => {
  if (Capacitor.isNativePlatform()) {
    try {
      await Filesystem.writeFile({
        path: fileName,
        data: data,
        directory: Directory.Documents,
        encoding: isBase64 ? undefined : Encoding.UTF8,
      });
      alert(`File saved to Documents/${fileName}`);
    } catch (e) {
      console.error('Error saving file:', e);
      alert('Failed to save file. Please check permissions.');
    }
  } else {
    // Web fallback
    const blob = isBase64
      ? await (await fetch(`data:${contentType};base64,${data}`)).blob()
      : new Blob([data], { type: contentType });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
};

// ... (inside component)



interface MedicalRecord {
  id: string;
  date: string;
  type: 'X-Ray' | 'CT Scan' | 'MRI' | 'Blood Test';
  result: 'Normal' | 'Abnormal' | 'Requires Follow-up' | 'Pending';
  confidence: number;
  notes: string;
  doctorName?: string;
  imageUrl?: string;
  raw?: any;
}

interface MedicalHistoryProps {
  user: {
    username: string;
    firstName: string;
    lastName: string;
    email: string;
    role: string;
  };
  onNavigateBack: () => void;
}

export function MedicalHistory({ user, onNavigateBack }: MedicalHistoryProps) {
  const [records, setRecords] = useState<MedicalRecord[]>([]);
  const [filteredRecords, setFilteredRecords] = useState<MedicalRecord[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterResult, setFilterResult] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [viewing, setViewing] = useState<MedicalRecord | null>(null);

  // Load from Firestore
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        const historyRef = collection(db, 'medical-history');
        const q = query(historyRef, where('username', '==', user.username));
        const querySnapshot = await getDocs(q);

        const mapped: MedicalRecord[] = querySnapshot.docs.map(doc => {
          const r = doc.data();
          return {
            id: doc.id,
            date: r.date ? new Date(r.date).toISOString() : new Date().toISOString(),
            type: r.type || 'X-Ray',
            result: r.result || 'Pending',
            confidence: typeof r.confidence === 'number' ? r.confidence : 0,
            notes: r.notes || '',
            doctorName: r.doctorName || '',
            imageUrl: r.imageUrl || '',
            raw: r.raw,
          };
        });

        if (!cancelled) {
          setRecords(mapped);
          setFilteredRecords(mapped);
        }
      } catch (e: any) {
        if (!cancelled) {
          console.error("Error loading history:", e);
          setLoadError('Failed to load history from database');
          setRecords([]);
          setFilteredRecords([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [user.username]);

  // Filter records based on search and filters
  useEffect(() => {
    let filtered = [...records];

    if (searchTerm) {
      filtered = filtered.filter(record =>
        record.notes.toLowerCase().includes(searchTerm.toLowerCase()) ||
        record.doctorName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        record.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterType !== 'all') {
      filtered = filtered.filter(record => record.type === filterType);
    }

    if (filterResult !== 'all') {
      filtered = filtered.filter(record => record.result === filterResult);
    }

    setFilteredRecords(filtered);
  }, [searchTerm, filterType, filterResult, records]);

  const getResultColor = (result: string) => {
    switch (result) {
      case 'Normal': return 'bg-green-100 text-green-800';
      case 'Abnormal': return 'bg-red-100 text-red-800';
      case 'Requires Follow-up': return 'bg-yellow-100 text-yellow-800';
      case 'Pending': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 75) return 'text-yellow-600';
    if (confidence > 0) return 'text-red-600';
    return 'text-gray-400';
  };

  const downloadRecord = async (record: MedicalRecord) => {
    const payload = {
      username: user.username,
      date: record.date,
      type: record.type,
      result: record.result,
      confidence: record.confidence,
      notes: record.notes,
      doctorName: record.doctorName,
      imageUrl: record.imageUrl,
    };
    const fileName = `medical-record-${new Date(record.date).toISOString().slice(0, 10)}.json`;
    await saveFile(fileName, JSON.stringify(payload, null, 2), 'application/json');
  };

  const downloadImage = async (record: MedicalRecord) => {
    if (!record.imageUrl) return;
    try {
      const response = await fetch(record.imageUrl);
      const blob = await response.blob();
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64data = (reader.result as string).split(',')[1];
        const fileName = `xray-${new Date(record.date).toISOString().slice(0, 10)}.png`;
        await saveFile(fileName, base64data, 'image/png', true);
      };
      reader.readAsDataURL(blob);
    } catch (e) {
      console.error('Error downloading image:', e);
      alert('Failed to download image.');
    }
  };

  // Export a PDF report for a given record
  const downloadPDF = async (record: MedicalRecord) => {
    const doc = new jsPDF({ unit: 'pt', format: 'a4' });
    const margin = 40;
    let y = margin;

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);
    doc.text('LungDX Analysis Report', margin, y);
    y += 24;

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(12);
    doc.text(`User: ${user.username}`, margin, y); y += 18;
    doc.text(`Date: ${new Date(record.date).toLocaleString()}`, margin, y); y += 18;
    doc.text(`Type: ${record.type}`, margin, y); y += 18;
    doc.text(`Result: ${record.result}`, margin, y); y += 18;
    if (record.confidence > 0) { doc.text(`Confidence: ${record.confidence}%`, margin, y); y += 18; }
    if (record.doctorName) { doc.text(`Doctor: Dr. ${record.doctorName}`, margin, y); y += 18; }
    if (record.notes) {
      const split = doc.splitTextToSize(`Notes: ${record.notes}`, 515);
      doc.text(split, margin, y);
      y += split.length * 14 + 6;
    }

    const raw: any = (record as any).raw;
    if (raw) {
      if (raw.prediction) { doc.text(`Prediction: ${raw.prediction}`, margin, y); y += 18; }
      if (raw.riskLevel) { doc.text(`Risk Level: ${raw.riskLevel}`, margin, y); y += 18; }
      if (raw.detailedMetrics) {
        doc.text('Metrics:', margin, y); y += 16;
        const metrics = JSON.stringify(raw.detailedMetrics, null, 2);
        const mSplit = doc.splitTextToSize(metrics, 515);
        doc.setFont('courier', 'normal');
        doc.text(mSplit, margin, y);
        y += mSplit.length * 12 + 6;
        doc.setFont('helvetica', 'normal');
      }
    }

    if (record.imageUrl) {
      try {
        const imgBlob = await fetch(record.imageUrl).then(r => r.blob());
        const reader = new FileReader();
        const dataUrl: string = await new Promise((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(imgBlob);
        });
        const imgWidth = 515;
        const imgHeight = 300;
        if (y + imgHeight > 800) { doc.addPage(); y = margin; }
        doc.text('X-Ray Image:', margin, y); y += 14;
        doc.addImage(dataUrl, 'PNG', margin, y, imgWidth, imgHeight, undefined, 'FAST');
        y += imgHeight + 10;
      } catch {
        // ignore
      }
    }

    const fileName = `analysis-${new Date(record.date).toISOString().slice(0, 10)}.pdf`;

    if (Capacitor.isNativePlatform()) {
      const pdfBase64 = doc.output('datauristring').split(',')[1];
      await saveFile(fileName, pdfBase64, 'application/pdf', true);
    } else {
      doc.save(fileName);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4 pt-16">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">Medical History</h1>
            <p className="text-slate-600">Complete record of your medical analyses and tests</p>
          </div>
          <Button onClick={onNavigateBack} variant="outline" className="border-blue-200 text-blue-700 hover:bg-blue-50">
            Back to Dashboard
          </Button>
        </div>

        {/* Filters and Search */}
        <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-md">
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                  <Input
                    placeholder="Search by test type, doctor, or notes..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 border-blue-100 focus:border-blue-400"
                  />
                </div>
              </div>
              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-40 border-blue-100">
                  <Filter className="w-4 h-4 mr-2 text-slate-500" />
                  <SelectValue placeholder="Test Type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  <SelectItem value="X-Ray">X-Ray</SelectItem>
                  <SelectItem value="CT Scan">CT Scan</SelectItem>
                  <SelectItem value="MRI">MRI</SelectItem>
                  <SelectItem value="Blood Test">Blood Test</SelectItem>
                </SelectContent>
              </Select>
              <Select value={filterResult} onValueChange={setFilterResult}>
                <SelectTrigger className="w-40 border-blue-100">
                  <SelectValue placeholder="Result" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Results</SelectItem>
                  <SelectItem value="Normal">Normal</SelectItem>
                  <SelectItem value="Abnormal">Abnormal</SelectItem>
                  <SelectItem value="Requires Follow-up">Requires Follow-up</SelectItem>
                  <SelectItem value="Pending">Pending</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Statistics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-all">
            <CardContent className="pt-4 pb-4 px-2">
              <div className="flex flex-col items-center justify-center text-center">
                <div className="text-xl font-bold text-blue-600">{records.length}</div>
                <div className="text-xs text-slate-600">Total Tests</div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-all">
            <CardContent className="pt-4 pb-4 px-2">
              <div className="flex flex-col items-center justify-center text-center">
                <div className="text-xl font-bold text-teal-600">
                  {records.filter(r => r.result === 'Normal').length}
                </div>
                <div className="text-xs text-slate-600">Normal Results</div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-all">
            <CardContent className="pt-4 pb-4 px-2">
              <div className="flex flex-col items-center justify-center text-center">
                <div className="text-xl font-bold text-yellow-600">
                  {records.filter(r => r.result === 'Requires Follow-up').length}
                </div>
                <div className="text-xs text-slate-600">Follow-ups</div>
              </div>
            </CardContent>
          </Card>
          <Card className="bg-white/80 backdrop-blur-sm border-blue-100 shadow-sm hover:shadow-md transition-all">
            <CardContent className="pt-4 pb-4 px-2">
              <div className="flex flex-col items-center justify-center text-center">
                <div className="text-xl font-bold text-slate-600">
                  {records.filter(r => r.result === 'Pending').length}
                </div>
                <div className="text-xs text-slate-600">Pending</div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Records List */}
        <div className="space-y-4">
          {loading ? (
            <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
              <CardContent className="pt-6">
                <div className="text-center py-8">
                  <div className="w-12 h-12 mx-auto mb-4 rounded-full border-4 border-blue-200 border-t-blue-500 animate-spin" />
                  <h3 className="text-lg font-medium text-slate-900 mb-2">Loading records…</h3>
                </div>
              </CardContent>
            </Card>
          ) : filteredRecords.length === 0 ? (
            <Card className="bg-white/80 backdrop-blur-sm border-blue-100">
              <CardContent className="pt-6">
                <div className="text-center py-8">
                  <FileText className="w-12 h-12 mx-auto text-slate-400 mb-4" />
                  <h3 className="text-lg font-medium text-slate-900 mb-2">No records found</h3>
                  <p className="text-slate-600">{loadError ? 'Could not load your medical history.' : 'Try adjusting your search or filter criteria.'}</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            filteredRecords.map((record) => (
              <Card key={record.id} className="hover:shadow-lg transition-all bg-white/90 backdrop-blur-sm border-blue-50 hover:border-blue-200">
                <CardContent className="pt-6">
                  <div className="flex flex-col sm:flex-row items-start justify-between gap-4">
                    <div className="flex-1 w-full">
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-blue-700 border-blue-200 bg-blue-50">
                          {record.type}
                        </Badge>
                        <Badge className={getResultColor(record.result)}>
                          {record.result}
                        </Badge>
                        {record.confidence > 0 && (
                          <span className={`text-xs font-medium ${getConfidenceColor(record.confidence)}`}>
                            {record.confidence}%
                          </span>
                        )}
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600 mb-2">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3 text-slate-400" />
                          {new Date(record.date).toLocaleDateString()}
                        </div>
                        {record.doctorName && (
                          <>
                            <span>•</span>
                            <span>Dr. {record.doctorName}</span>
                          </>
                        )}
                      </div>

                      <p className="text-sm text-slate-700 mb-4 line-clamp-2">{record.notes}</p>
                    </div>

                    <div className="flex gap-2 w-full sm:w-auto">
                      <Button variant="outline" size="sm" onClick={() => setViewing(record)} className="flex-1 sm:flex-none border-blue-200 text-blue-700 hover:bg-blue-50">
                        <Eye className="w-4 h-4 mr-1" />
                        View
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => downloadPDF(record)} className="flex-1 sm:flex-none border-blue-200 text-blue-700 hover:bg-blue-50">
                        <Download className="w-4 h-4 mr-1" />
                        PDF
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => downloadRecord(viewing || record)} className="flex-1 sm:flex-none border-blue-200 text-blue-700 hover:bg-blue-50">
                        <Download className="w-4 h-4 mr-1" />
                        JSON
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </div>

        {/* View Dialog */}
        <Dialog open={!!viewing} onOpenChange={(open) => !open && setViewing(null)}>
          <DialogContent className="max-w-3xl bg-white/95 backdrop-blur-sm border-blue-100">
            <DialogHeader>
              <DialogTitle className="text-xl font-bold text-slate-900">Analysis Details</DialogTitle>
            </DialogHeader>
            {viewing && (
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <Badge variant="outline" className="text-blue-700 border-blue-200 bg-blue-50">{viewing.type}</Badge>
                  <Badge className={getResultColor(viewing.result)}>{viewing.result}</Badge>
                  {viewing.confidence > 0 && (
                    <span className={`text-sm font-medium ${getConfidenceColor(viewing.confidence)}`}>
                      {viewing.confidence}% confidence
                    </span>
                  )}
                </div>
                <div className="text-sm text-slate-600 flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-slate-400" />
                  {new Date(viewing.date).toLocaleString()}
                  {viewing.doctorName && <><span>•</span><span>Dr. {viewing.doctorName}</span></>}
                </div>
                {viewing.imageUrl ? (
                  <div className="relative rounded-lg overflow-hidden bg-slate-100 border border-slate-200">
                    <img src={viewing.imageUrl} alt="X-ray" className="w-full h-auto max-h-[60vh] object-contain" />
                  </div>
                ) : (
                  <p className="text-sm text-slate-600">No image available for this record.</p>
                )}
                {viewing.notes && <p className="text-slate-700">{viewing.notes}</p>}
                {viewing.raw && (
                  <div className="rounded-md border border-blue-100 p-3 bg-blue-50/50">
                    <h4 className="font-medium mb-2 text-slate-900">Analysis Details</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                      {viewing.raw.prediction && (
                        <div>
                          <span className="text-slate-500">Prediction: </span>
                          <span className="font-medium text-slate-900">{viewing.raw.prediction}</span>
                        </div>
                      )}
                      {viewing.raw.riskLevel && (
                        <div>
                          <span className="text-slate-500">Risk Level: </span>
                          <span className="font-medium text-slate-900">{viewing.raw.riskLevel}</span>
                        </div>
                      )}
                      {viewing.raw.detailedMetrics && (
                        <div className="sm:col-span-2">
                          <span className="text-slate-500">Metrics: </span>
                          <pre className="mt-1 text-xs bg-white p-2 rounded-md overflow-x-auto border border-blue-100 text-slate-700">{JSON.stringify(viewing.raw.detailedMetrics, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  </div>
                )}
                <div className="flex justify-end gap-2">
                  <Button variant="outline" onClick={() => downloadRecord(viewing)} className="border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Download className="w-4 h-4 mr-1" /> Download JSON
                  </Button>
                  {viewing.imageUrl && (
                    <Button variant="outline" onClick={() => downloadImage(viewing)} className="border-blue-200 text-blue-700 hover:bg-blue-50">
                      <Download className="w-4 h-4 mr-1" /> Download Image
                    </Button>
                  )}
                  <Button variant="outline" onClick={() => downloadPDF(viewing)} className="border-blue-200 text-blue-700 hover:bg-blue-50">
                    <Download className="w-4 h-4 mr-1" /> Download PDF
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}