import React, { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { ArrowLeft, Loader2 } from 'lucide-react';
import { BACKEND_BASE } from '../utils/config';

interface StaticPageProps {
  pageType: 'aboutUs' | 'contact' | 'terms';
  onNavigateBack: () => void;
}

interface PageData {
  title: string;
  content: string;
  images: string[];
}

export function StaticPage({ pageType, onNavigateBack }: StaticPageProps) {
  const [pageData, setPageData] = useState<PageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchPageData = async () => {
      try {
        const response = await fetch(`${BACKEND_BASE}/api/static/${pageType}`, {
          headers: {
            // 'Authorization': `Bearer ${publicAnonKey}` // Removed
          }
        });

        const data = await response.json();

        if (data.success) {
          setPageData(data.data);
        } else {
          setError('Failed to load page content');
        }
      } catch (error) {
        console.error('Error fetching page data:', error);
        setError('Network error. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchPageData();
  }, [pageType]);

  const getPageTitle = () => {
    switch (pageType) {
      case 'aboutUs':
        return 'About Us';
      case 'contact':
        return 'Contact Us';
      case 'terms':
        return 'Terms and Conditions';
      default:
        return 'Page';
    }
  };

  const getPageIcon = () => {
    switch (pageType) {
      case 'aboutUs':
        return '🏥';
      case 'contact':
        return '📞';
      case 'terms':
        return '📋';
      default:
        return '📄';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-slate-600">Loading page content...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 flex items-center justify-center p-4">
        <div className="text-center">
          <div className="text-6xl mb-4">⚠️</div>
          <h2 className="text-2xl font-semibold mb-4 text-slate-900">Unable to Load Page</h2>
          <p className="text-red-600 mb-4">{error}</p>
          <p className="text-slate-600">Please use the navigation above to go back or try refreshing the page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-teal-50 p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center pt-8">
          <div className="text-4xl mb-2">{getPageIcon()}</div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-teal-600 bg-clip-text text-transparent">{getPageTitle()}</h1>
        </div>

        {/* Content Card */}
        <Card className="shadow-lg bg-white/80 backdrop-blur-sm border-blue-100">
          <CardHeader>
            <CardTitle className="text-2xl text-center text-slate-800">
              {pageData?.title || getPageTitle()}
            </CardTitle>
          </CardHeader>
          <CardContent className="prose max-w-none text-slate-600">
            {pageType === 'aboutUs' && (
              <div className="space-y-6">
                <div className="text-center">
                  <div className="text-6xl mb-4">🫁</div>
                  <h2 className="text-2xl font-semibold mb-4 text-slate-800">Advanced Lung Diagnosis with LungDx😊</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div className="text-center p-4 bg-blue-50 rounded-xl border border-blue-100">
                    <div className="text-3xl mb-2">🤖</div>
                    <h3 className="font-semibold mb-2 text-blue-900">AI-Powered</h3>
                    <p className="text-sm text-slate-600">Advanced machine learning algorithms for accurate detection</p>
                  </div>
                  <div className="text-center p-4 bg-teal-50 rounded-xl border border-teal-100">
                    <div className="text-3xl mb-2">⚡</div>
                    <h3 className="font-semibold mb-2 text-teal-900">Fast Results</h3>
                    <p className="text-sm text-slate-600">Quick analysis and immediate results for timely intervention</p>
                  </div>
                  <div className="text-center p-4 bg-indigo-50 rounded-xl border border-indigo-100">
                    <div className="text-3xl mb-2">🎯</div>
                    <h3 className="font-semibold mb-2 text-indigo-900">High Accuracy</h3>
                    <p className="text-sm text-slate-600">State-of-the-art precision in medical image analysis</p>
                  </div>
                </div>

                <div className="text-slate-700 leading-relaxed space-y-4">
                  <p>
                    {pageData?.content || "We are dedicated to advancing lung diagnosis through innovative AI technology. Our mission is to provide early, accurate detection to save lives with LungDx😊."}
                  </p>
                  <p>
                    Our platform utilizes cutting-edge deep learning models trained on extensive medical datasets to analyze chest X-rays and CT scans. We work closely with medical professionals to ensure our technology meets the highest standards of accuracy and reliability.
                  </p>
                  <p>
                    Early detection of lung cancer significantly improves patient outcomes. Our goal is to make advanced diagnostic tools accessible to healthcare providers worldwide, enabling faster diagnosis and better patient care.
                  </p>
                </div>
              </div>
            )}

            {pageType === 'contact' && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <div className="text-6xl mb-4">📞</div>
                  <h2 className="text-2xl font-semibold mb-4 text-slate-800">Get In Touch</h2>
                  <p className="text-slate-600">We're here to help with any questions or support needs</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="p-6 bg-blue-50 rounded-xl border border-blue-100">
                    <h3 className="font-semibold mb-3 flex items-center text-blue-900">
                      📧 Email Support
                    </h3>
                    <p className="text-slate-700 mb-2">For general inquiries and support:</p>
                    <p className="text-blue-600 font-medium">support@lungcancer.com</p>
                  </div>

                  <div className="p-6 bg-teal-50 rounded-xl border border-teal-100">
                    <h3 className="font-semibold mb-3 flex items-center text-teal-900">
                      📱 Phone Support
                    </h3>
                    <p className="text-slate-700 mb-2">24/7 technical support:</p>
                    <p className="text-teal-600 font-medium">+1-555-0123</p>
                  </div>

                  <div className="p-6 bg-indigo-50 rounded-xl border border-indigo-100">
                    <h3 className="font-semibold mb-3 flex items-center text-indigo-900">
                      🏥 Medical Inquiries
                    </h3>
                    <p className="text-slate-700 mb-2">For healthcare professionals:</p>
                    <p className="text-indigo-600 font-medium">medical@lungcancer.com</p>
                  </div>

                  <div className="p-6 bg-orange-50 rounded-xl border border-orange-100">
                    <h3 className="font-semibold mb-3 flex items-center text-orange-900">
                      🚨 Emergency
                    </h3>
                    <p className="text-slate-700 mb-2">Critical system issues:</p>
                    <p className="text-orange-600 font-medium">emergency@lungcancer.com</p>
                  </div>
                </div>

                <div className="mt-8 p-6 border border-blue-100 rounded-xl bg-white/50">
                  <h3 className="font-semibold mb-3 text-slate-800">Office Hours</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium text-slate-900">Monday - Friday</p>
                      <p className="text-slate-600">9:00 AM - 6:00 PM EST</p>
                    </div>
                    <div>
                      <p className="font-medium text-slate-900">Weekend Support</p>
                      <p className="text-slate-600">Emergency cases only</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {pageType === 'terms' && (
              <div className="space-y-6">
                <div className="text-center mb-8">
                  <div className="text-6xl mb-4">📋</div>
                  <h2 className="text-2xl font-semibold mb-4 text-slate-800">Terms and Conditions</h2>
                  <p className="text-slate-600">Please read these terms carefully before using our service</p>
                </div>

                <div className="space-y-6 text-slate-700">
                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">1. Acceptance of Terms</h3>
                    <p>{pageData?.content || "By using this service, you agree to our terms of service and privacy policy. Please read carefully before proceeding."}</p>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">2. Medical Disclaimer</h3>
                    <p>This platform is designed to assist healthcare professionals in medical diagnosis. It is not intended to replace professional medical judgment or advice. Always consult with qualified healthcare providers for medical decisions.</p>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">3. User Responsibilities</h3>
                    <ul className="list-disc list-inside space-y-1 ml-4">
                      <li>Provide accurate and complete information during registration</li>
                      <li>Maintain the security of your account credentials</li>
                      <li>Use the platform only for legitimate medical purposes</li>
                      <li>Comply with all applicable laws and regulations</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">4. Privacy and Data Protection</h3>
                    <p>We are committed to protecting your privacy and medical data in accordance with HIPAA and other applicable regulations. All patient data is encrypted and stored securely.</p>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">5. Limitation of Liability</h3>
                    <p>While we strive for accuracy, no medical diagnostic tool is 100% accurate. Users acknowledge that they use this service at their own risk and we are not liable for any medical decisions based on our analysis.</p>
                  </div>

                  <div>
                    <h3 className="font-semibold mb-3 text-slate-900">6. Updates to Terms</h3>
                    <p>We reserve the right to update these terms at any time. Users will be notified of significant changes and continued use constitutes acceptance of updated terms.</p>
                  </div>

                  <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                    <h3 className="font-semibold mb-2 text-yellow-800">Important Notice</h3>
                    <p className="text-yellow-700">This AI diagnostic tool is intended for use by licensed healthcare professionals only. Patients should always consult with their healthcare provider for medical advice and treatment decisions.</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}