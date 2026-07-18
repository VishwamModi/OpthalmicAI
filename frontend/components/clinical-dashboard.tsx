'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Eye, LogOut, Printer, Sparkles, Trash2 } from 'lucide-react';
import { apiUrl } from '@/lib/api-base';

import { PrintReport } from './dashboard/print-report';
export { ClinicalDashboard } from './clinical-dashboard-scroll';

interface ClinicalDashboardProps {
  onLogout: () => void;
}

type AuthSession = {
  user?: { email?: string; role?: string; name?: string } | null;
  patient_id?: number | null;
  patient_uid?: string | null;
  patient_name?: string | null;
  patient_dob?: string | null;
  doctor_name?: string | null;
};

type ScanResults = {
  scan_id?: number;
  patient_id?: number;
  patient_uid?: string;
  patient_name?: string;
  patient_dob?: string;
  doctor_name?: string;
  diagnosis: string;
  icd10: string;
  etdrsLevel: string;
  severity: number;
  maxSeverity: number;
  aiConfidence: number; // already 0–100
  processingTime?: string;
  modelVersion?: string;
  recommendation: string;
  generatedReport?: string; // backward compatibility
  clinical_report?: string;
  patient_report?: string;
  gradcam_base64?: string;
  clinicalFeatures: Array<{
    name: string;
    value: string;
    unit?: string;
    reference?: string;
    status: 'normal' | 'warning' | 'elevated' | 'critical';
    score: number; // 0–1
  }>;
  classProbs: Record<string, number>; // 0–1
  severityStages: Array<{
    level: number;
    label: string;
    description: string;
    current?: boolean;
  }>;
};

function EmptyState({ title }: { title: string }) {
  return (
    <div className="rounded-lg border border-[#E5DDD6] bg-white p-10 text-center">
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-lg bg-[#F7F5F0] border border-[#E5DDD6]">
        <Eye className="h-6 w-6 text-[#A85D4A]" />
      </div>
      <h3 className="text-lg font-semibold text-[#2C2825]">{title}</h3>
      <p className="mt-2 text-sm text-[#736D68]">
        Please upload a retinal fundus scan to view clinical insights.
      </p>
    </div>
  );
}

function ScanningOverlay({ stageText }: { stageText: string }) {
  return (
    <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
      <div
        className="absolute inset-0"
        style={{
          background: `linear-gradient(to bottom,
            transparent 0%,
            rgba(168, 93, 74, 0.25) 48%,
            rgba(168, 93, 74, 0.55) 50%,
            rgba(168, 93, 74, 0.25) 52%,
            transparent 100%)`,
          animation: 'scan 2s ease-in-out infinite',
        }}
      />
      <div className="absolute inset-x-0 bottom-3 flex items-center justify-between px-3">
        <div className="rounded-full bg-white/85 px-3 py-1 text-[11px] font-semibold text-[#2C2825] border border-[#E5DDD6]">
          {stageText}
        </div>
        <div className="inline-flex items-center gap-2 rounded-full bg-white/85 px-3 py-1 text-[11px] text-[#736D68] border border-[#E5DDD6]">
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[#E5DDD6] border-t-[#A85D4A]" />
          Processing…
        </div>
      </div>

      <style>{`
        @keyframes scan {
          0% { transform: translateY(-35%); }
          50% { transform: translateY(0); }
          100% { transform: translateY(120%); }
        }
      `}</style>
    </div>
  );
}

export function ClinicalDashboard({ onLogout }: ClinicalDashboardProps) {
  const [activeTab, setActiveTab] = useState<'upload' | 'overview' | 'features' | 'staging' | 'dca'>('upload');
  const inputRef = useRef<HTMLInputElement>(null);

  const [scanResults, setScanResults] = useState<ScanResults | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reportType, setReportType] = useState<'clinical' | 'patient'>('clinical');
  const [showAiReport, setShowAiReport] = useState(false);
  const [isChatBotOpen, setIsChatBotOpen] = useState(false);
  const [hasAutoOpenedChatBot, setHasAutoOpenedChatBot] = useState(false);

  const [messages, setMessages] = useState<Array<{ role: 'user' | 'ai'; content: string }>>([]);
  const [input, setInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const [heatmapMode, setHeatmapMode] = useState<'heatmap' | 'original'>('heatmap');
  const [authSession, setAuthSession] = useState<AuthSession | null>(null);

  const processingStages = useMemo(
    () => [
      'Applying fundus mask + CLAHE…',
      'Running EfficientNet‑B3 ensembles…',
      'Applying clinical thresholds…',
      'Compiling clinical report…',
    ],
    []
  );
  const [processingStage, setProcessingStage] = useState(0);

  const tabs = [
    { id: 'upload', label: 'Upload Scan' },
    { id: 'overview', label: 'Overview' },
    { id: 'features', label: 'Clinical Features' },
    { id: 'staging', label: 'Severity Staging' },
    { id: 'dca', label: 'Decision Curve' },
  ];

  useEffect(() => {
    try {
      const raw = localStorage.getItem('opthalmic_session');
      if (!raw) return;
      const parsed = JSON.parse(raw) as AuthSession;
      setAuthSession(parsed);
    } catch {}
  }, []);

  useEffect(() => {
    if (!isProcessing) return;
    setProcessingStage(0);
    const t = setInterval(() => {
      setProcessingStage((prev) => Math.min(prev + 1, processingStages.length - 1));
    }, 850);
    return () => clearInterval(t);
  }, [isProcessing, processingStages.length]);

  const handlePrint = () => {
    window.print();
  };

  const handleImageUpload = async (file: File) => {
    setError(null);
    setScanResults(null);

    const url = URL.createObjectURL(file);
    setSelectedImage(url);

    setIsProcessing(true);
    try {
      const fd = new FormData();
      fd.append('file', file);

      const res = await fetch(apiUrl('/api/analyze'), {
        method: 'POST',
        headers: authSession?.user?.email ? { 'X-User-Email': authSession.user.email } : undefined,
        body: fd,
      });
      const json = (await res.json()) as any;
      if (!res.ok || json?.status !== 'success') {
        throw new Error(json?.message || 'Analysis failed');
      }
      const data = json?.data;
      if (!data?.diagnosis) {
        throw new Error('Malformed response from server (missing diagnosis).');
      }
      setScanResults(data as ScanResults);
      setActiveTab('overview');
    } catch (e: any) {
      setError(e?.message || 'Upload failed');
      // If the backend validation fails (e.g., dog/selfie), show that message here.
      setActiveTab('upload');
    } finally {
      setIsProcessing(false);
    }
  };

  const severityScore = useMemo(() => {
    if (!scanResults) return 0;
    const denom = Number(scanResults.maxSeverity || 5);
    const num = Number(scanResults.severity || 0);
    if (!denom) return 0;
    return Math.max(0, Math.min(100, (num / denom) * 100));
  }, [scanResults]);

  const narrativeParagraphs = useMemo(() => {
    const chosen =
      reportType === 'clinical'
        ? scanResults?.clinical_report || scanResults?.generatedReport || ''
        : scanResults?.patient_report || '';
    const txt = (chosen || '').trim();
    if (!txt) return [];
    return txt
      .split(/\n+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }, [scanResults, reportType]);

  const fallbackNarrativeParagraphs = useMemo(() => {
    if (!scanResults) return [];
    const clinicalFeaturesTop = (scanResults.clinicalFeatures || [])
      .slice(0, 3)
      .map((f) => `${f.name}: ${f.value}${f.unit ? ` ${f.unit}` : ''}`)
      .join(' • ');

    if (reportType === 'clinical') {
      return [
        `Diagnosis: ${scanResults.diagnosis} (${scanResults.etdrsLevel}). ICD-10: ${scanResults.icd10}.`,
        `The uncalibrated ordinal-proximity indicator is ${scanResults.aiConfidence.toFixed(1)}% with severity score ${severityScore.toFixed(0)}%.`,
        clinicalFeaturesTop
          ? `Key extracted features: ${clinicalFeaturesTop}.`
          : 'Feature extraction was completed; detailed metrics are available in Clinical Features.',
        `Recommendation: ${scanResults.recommendation}`,
      ];
    }

    return [
      `Your eye scan suggests: ${scanResults.diagnosis}.`,
      `Ordinal proximity is ${scanResults.aiConfidence.toFixed(1)}%. This is not a calibrated probability and is not a diagnosis.`,
      `Recommended next step: ${scanResults.recommendation}`,
    ];
  }, [scanResults, reportType, severityScore]);

  const gradcamSrc = useMemo(() => {
    const b64 = scanResults?.gradcam_base64;
    if (!b64) return null;
    return `data:image/png;base64,${b64}`;
  }, [scanResults]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

  useEffect(() => {
    if (scanResults) {
      setShowAiReport(true);
    }
  }, [scanResults]);

  const sendChat = async (text: string) => {
    const message = text.trim();
    if (!message) return;
    const scanId = scanResults?.scan_id;
    if (!scanId) {
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: 'Please analyze a scan first so I can answer with your specific context.' },
      ]);
      return;
    }

    setMessages((prev) => [...prev, { role: 'user', content: message }]);
    setChatLoading(true);
    try {
      const res = await fetch(apiUrl('/api/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_id: scanId, message }),
      });
      const json = (await res.json()) as any;
      if (!res.ok || !json?.reply) {
        throw new Error(json?.detail || json?.message || 'Chat request failed');
      }
      setMessages((prev) => [...prev, { role: 'ai', content: String(json.reply) }]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'ai', content: e?.message || 'Unable to reach the chat service right now.' },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const reportPatient = useMemo(() => {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');

    const resolvedDob = scanResults?.patient_dob || authSession?.patient_dob || '';
    let resolvedAge: number | undefined;
    if (resolvedDob) {
      const dobDate = new Date(resolvedDob);
      if (!Number.isNaN(dobDate.getTime())) {
        let age = today.getFullYear() - dobDate.getFullYear();
        const hasBirthdayPassed =
          today.getMonth() > dobDate.getMonth() ||
          (today.getMonth() === dobDate.getMonth() && today.getDate() >= dobDate.getDate());
        if (!hasBirthdayPassed) age -= 1;
        resolvedAge = Math.max(age, 0);
      }
    }

    return {
      id: scanResults?.patient_uid || authSession?.patient_uid || '—',
      name: scanResults?.patient_name || authSession?.patient_name || '—',
      dob: resolvedDob || '—',
      age: resolvedAge ?? '—',
      eye: 'Right Eye (OD)',
      examDate: `${yyyy}-${mm}-${dd}`,
      referringPhysician:
        scanResults?.doctor_name ||
        authSession?.doctor_name ||
        authSession?.user?.email ||
        '—',
    };
  }, [scanResults, authSession]);

  return (
    <div className="min-h-screen bg-[#FCF8F3]">
      <PrintReport
        patient={reportPatient as any}
        results={scanResults as any}
        originalImageSrc={selectedImage}
        gradcamImageSrc={gradcamSrc}
      />
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-[#E8D9C8] bg-[#FCF8F3] print:hidden">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Logo + Patient Info */}
            <div className="flex items-center gap-6">
              <div className="flex items-center gap-3">
                <img src="/images/logo.jpeg" alt="Opthalmic Logo" className="h-16 w-auto object-contain" />
              </div>
              
              <div className="border-l border-[#E8D9C8] pl-6">
                <p className="text-xs text-[#6E4B34]">Patient ID</p>
                <p className="font-semibold text-[#3B2416]">
                  {scanResults?.patient_uid || authSession?.patient_uid || '—'}
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3">
              <button
                onClick={handlePrint}
                disabled={!scanResults}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#3B2416] hover:bg-[#2F1B11] disabled:opacity-50 text-white text-sm font-medium transition-colors"
              >
                <Printer className="w-4 h-4" />
                Export PDF
              </button>
              <button
                onClick={() => {
                  try {
                    localStorage.removeItem('opthalmic_session');
                  } catch {}
                  onLogout();
                }}
                className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#D6C3AE] text-[#3B2416] hover:bg-[#F4EADD] text-sm font-medium transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </button>
            </div>
          </div>

          {/* Tabs */}
          <div className="mt-6 flex gap-1 border-b border-[#E8D9C8]">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-4 py-3 border-b-2 transition-colors text-sm font-medium ${
                  activeTab === tab.id
                    ? 'border-[#3B2416] text-[#3B2416]'
                    : 'border-transparent text-[#6E4B34] hover:text-[#3B2416]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="mx-auto max-w-7xl px-6 py-8">
        {activeTab === 'upload' && (
          <div className="max-w-3xl mx-auto py-4 flex flex-col gap-6">
            <div className="rounded-lg border border-[#E8D9C8] bg-white p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-[#3B2416]">Upload retinal fundus scan</h2>
                  <p className="text-sm text-[#6E4B34] mt-1">
                    Drag & drop or click to browse. Supported: JPG/PNG.
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-[#F8F2EA] text-[#6E4B34] text-xs font-medium border border-[#E8D9C8]">
                  Local processing
                </span>
              </div>

              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleImageUpload(f);
                }}
              />

              <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  const f = e.dataTransfer.files?.[0];
                  if (f) void handleImageUpload(f);
                }}
                className="mt-5 relative cursor-pointer rounded-lg border-2 border-dashed border-[#D6C3AE] bg-white hover:bg-[#F8F2EA] transition-colors p-5"
              >
                <div className="min-h-[280px] grid place-items-center text-center">
                  {selectedImage ? (
                    <div className="w-full">
                      <div className="relative overflow-hidden rounded-lg border border-[#E5E5E5] bg-white">
                        <img src={selectedImage} alt="Preview" className="h-72 w-full object-contain" />
                        {isProcessing ? (
                          <ScanningOverlay stageText={processingStages[processingStage]} />
                        ) : null}
                      </div>
                      <div className="mt-3 text-xs text-[#666666]">
                        {isProcessing
                          ? 'Processing… please keep this tab open.'
                          : 'Click again to upload a different scan.'}
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-sm">
                      <div className="text-4xl mb-3">📷</div>
                      <p className="text-base font-semibold text-[#3B2416]">Drag & drop or click to browse</p>
                      <p className="text-sm text-[#6E4B34] mt-1">
                        Upload a retinal fundus scan to generate clinical insights.
                      </p>
                      <div className="flex items-center justify-center gap-2 mt-4">
                        {['PNG', 'JPG'].map((fmt) => (
                          <span
                            key={fmt}
                            className="px-2 py-0.5 rounded-full bg-[#F3F3F3] text-xs font-medium text-[#666666] border border-[#E5E5E5]"
                          >
                            {fmt}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {error ? (
                <div className="mt-5 rounded-lg border border-[#E8C8BE] bg-[#FCF3F0] px-4 py-3 text-sm text-[#A85D4A]">
                  {error}
                </div>
              ) : null}
            </div>
          </div>
        )}

        {activeTab === 'overview' && (
          scanResults ? (
            <div className="flex flex-col gap-6">
              <div className="rounded-lg border border-[#E5DDD6] bg-white p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="flex flex-col gap-1">
                  <div className="text-sm text-[#736D68] font-medium">Primary Diagnosis</div>
                  <h2 className="text-2xl font-bold text-[#2C2825]">{scanResults.diagnosis}</h2>
                  <div className="flex flex-wrap gap-2 mt-1">
                    <span className="px-2.5 py-1 rounded-lg bg-[#A85D4A]/10 text-[#A85D4A] text-xs font-semibold border border-[#A85D4A]/20">
                      {scanResults.etdrsLevel}
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-[#F7F5F0] text-[#736D68] text-xs font-medium border border-[#E5DDD6]">
                      ICD-10: {scanResults.icd10}
                    </span>
                  </div>
                </div>

                <div className="flex flex-col items-center gap-1 min-w-[120px]">
                  <div className="relative w-24 h-24">
                    <svg viewBox="0 0 36 36" className="w-24 h-24 -rotate-90">
                      <circle cx="18" cy="18" r="15.9" fill="none" stroke="#E5DDD6" strokeWidth="2.8" />
                      <circle
                        cx="18" cy="18" r="15.9" fill="none"
                        stroke="#A85D4A" strokeWidth="2.8"
                        strokeDasharray={`${scanResults.aiConfidence} ${100 - scanResults.aiConfidence}`}
                        strokeLinecap="round"
                      />
                    </svg>
                    <div className="absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-xl font-bold text-[#2C2825]">{scanResults.aiConfidence.toFixed(1)}%</span>
                    </div>
                  </div>
                  <span className="text-xs text-[#736D68] font-medium">Model Confidence</span>
                </div>
              </div>

              {/* Severity (full-width, above images) */}
              <div className="rounded-lg border border-[#E5DDD6] bg-white p-5">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
                  <div>
                    <h3 className="font-semibold text-[#2C2825]">Severity score</h3>
                    <p className="text-xs text-[#736D68] mt-0.5">Computed as (severity / maxSeverity) × 100.</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-xs text-[#736D68] font-mono">
                      {severityScore.toFixed(0)}%
                    </div>
                    <div className="w-56 h-2 rounded-full bg-[#E5DDD6] overflow-hidden">
                      <div className="h-full bg-[#A85D4A]" style={{ width: `${severityScore}%` }} />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="rounded-lg border border-[#E5DDD6] bg-[#F7F5F0] px-3 py-2 text-xs text-[#736D68]">
                    <div className="text-[11px]">Scan ID</div>
                    <div className="font-mono text-[#2C2825] text-sm">#{scanResults.scan_id ?? '—'}</div>
                  </div>
                  <div className="rounded-lg border border-[#E5DDD6] bg-[#F7F5F0] px-3 py-2 text-xs text-[#736D68]">
                    <div className="text-[11px]">Patient</div>
                    <div className="font-mono text-[#2C2825] text-sm">{scanResults.patient_uid ?? '—'}</div>
                  </div>
                  <div className="rounded-lg border border-[#E5DDD6] bg-[#F7F5F0] px-3 py-2 text-xs text-[#736D68]">
                    <div className="text-[11px]">ETDRS</div>
                    <div className="text-[#2C2825] text-sm font-semibold">{scanResults.etdrsLevel}</div>
                  </div>
                </div>
              </div>

              {/* Grad-CAM (full-width, below severity) */}
              <div className="rounded-lg border border-[#E5DDD6] bg-white p-5">
                <div className="flex items-center justify-between gap-3 mb-3">
                  <div>
                    <h3 className="font-semibold text-[#2C2825]">Grad‑CAM</h3>
                    <span className="text-xs text-[#736D68]">Explainability heatmap</span>
                  </div>
                  <div className="inline-flex rounded-lg border border-[#E5DDD6] bg-white p-1 print:hidden">
                    <button
                      onClick={() => setHeatmapMode('heatmap')}
                      className={[
                        'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                        heatmapMode === 'heatmap'
                          ? 'bg-[#A85D4A] text-white'
                          : 'text-[#736D68] hover:text-[#2C2825]',
                      ].join(' ')}
                    >
                      Heatmap
                    </button>
                    <button
                      onClick={() => setHeatmapMode('original')}
                      className={[
                        'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                        heatmapMode === 'original'
                          ? 'bg-[#A85D4A] text-white'
                          : 'text-[#736D68] hover:text-[#2C2825]',
                      ].join(' ')}
                    >
                      Original
                    </button>
                  </div>
                </div>

                <div className="relative overflow-hidden rounded-lg border border-[#E5DDD6] bg-[#F7F5F0]">
                  <div className="aspect-[16/9] w-full" />
                  {heatmapMode === 'heatmap' && gradcamSrc ? (
                    <img
                      src={gradcamSrc}
                      alt="Grad-CAM heatmap"
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Retinal scan"
                      className="absolute inset-0 h-full w-full object-contain"
                    />
                  ) : (
                    <div className="absolute inset-0 grid place-items-center text-xs text-[#A39A8E]">
                      Upload a scan to view images.
                    </div>
                  )}
                  {heatmapMode === 'heatmap' && !gradcamSrc ? (
                    <div className="absolute inset-0 grid place-items-center text-xs text-[#A39A8E]">
                      Heatmap is unavailable (Grad‑CAM not generated).
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="rounded-lg border border-[#E5DDD6] bg-white p-5">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                  <div>
                    <h3 className="font-semibold text-[#2C2825]">AI Report</h3>
                  </div>
                  <div className="flex items-center gap-2 print:hidden">
                    <button
                      onClick={() => {
                        setShowAiReport((v) => {
                          const next = !v;
                          if (next && !hasAutoOpenedChatBot) {
                            setIsChatBotOpen(true);
                            setHasAutoOpenedChatBot(true);
                          }
                          return next;
                        });
                      }}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-[#A85D4A] hover:bg-[#935246] text-white text-xs font-semibold transition-colors"
                    >
                      {showAiReport ? 'Hide AI Report' : 'Show AI Report'}
                    </button>
                    <button
                      onClick={handlePrint}
                      disabled={!scanResults}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-[#E5DDD6] text-[#2C2825] hover:bg-[#F7F5F0] disabled:opacity-50 text-xs font-semibold transition-colors"
                    >
                      🖨️ Print Report
                    </button>
                  </div>
                </div>

                {showAiReport ? (
                  <div className="mt-4 rounded-xl border border-[#E8C8BE] bg-gradient-to-b from-[#FCF3F0] to-white p-4 shadow-sm">
                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-3">
                      <div className="inline-flex rounded-lg border border-[#E5DDD6] bg-white p-1 w-fit">
                        <button
                          onClick={() => setReportType('clinical')}
                          className={[
                            'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                            reportType === 'clinical'
                              ? 'bg-[#A85D4A] text-white'
                              : 'text-[#736D68] hover:text-[#2C2825]',
                          ].join(' ')}
                        >
                          Clinical View
                        </button>
                        <button
                          onClick={() => setReportType('patient')}
                          className={[
                            'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                            reportType === 'patient'
                              ? 'bg-[#A85D4A] text-white'
                              : 'text-[#736D68] hover:text-[#2C2825]',
                          ].join(' ')}
                        >
                          Patient View
                        </button>
                      </div>
                    </div>

                    {(narrativeParagraphs.length > 0 ? narrativeParagraphs : fallbackNarrativeParagraphs).length > 0 ? (
                      <div className="rounded-lg border border-[#E5DDD6] bg-white p-4">
                        {(narrativeParagraphs.length > 0 ? narrativeParagraphs : fallbackNarrativeParagraphs).map((p, idx) => (
                          <p key={idx} className="text-sm text-[#5A5754] leading-7 mb-4 last:mb-0">
                            {p}
                          </p>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-[#A39A8E] leading-relaxed">Preparing report...</p>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          ) : (
            <EmptyState title="Overview" />
          )
        )}

        {activeTab === 'features' && (
          scanResults ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(scanResults.clinicalFeatures || []).map((f) => {
                const statusStyles: Record<string, { border: string; bg: string; text: string }> = {
                  normal: { border: '#D4CCC2', bg: '#F0F4EE', text: '#6B705C' },
                  warning: { border: '#ECC9B0', bg: '#FDF8F4', text: '#CB997E' },
                  elevated: { border: '#E8C8BE', bg: '#FCF3F0', text: '#A85D4A' },
                  critical: { border: '#E8B5B6', bg: '#FCF0EF', text: '#9E2A2B' },
                };
                const s = statusStyles[f.status] ?? statusStyles.normal;
                return (
                  <div
                    key={f.name}
                    className="rounded-lg border bg-white p-5 flex flex-col gap-3 transition-all hover:shadow-sm"
                    style={{ borderColor: s.border }}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <h3 className="font-semibold text-sm text-[#2C2825]">{f.name}</h3>
                        <p className="text-lg font-bold mt-0.5" style={{ color: s.text }}>
                          {f.value}
                          {f.unit ? <span className="text-sm font-normal text-[#736D68] ml-1">{f.unit}</span> : null}
                        </p>
                      </div>
                      <span
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border shrink-0"
                        style={{ color: s.text, background: s.bg, borderColor: s.border }}
                      >
                        {f.status}
                      </span>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs text-[#736D68] mb-1">
                        <span>Severity score</span>
                        <span className="font-semibold" style={{ color: s.text }}>
                          {(Number(f.score) * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-2 rounded-full bg-[#E5DDD6] overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.max(0, Math.min(100, Number(f.score) * 100))}%`, background: s.text }}
                        />
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-xs text-[#736D68] pt-1 border-t border-[#E5DDD6]">
                      <span>
                        Reference:{' '}
                        <strong className="text-[#2C2825]">{f.reference ?? '—'}</strong>
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <EmptyState title="Clinical Features" />
          )
        )}

        {activeTab === 'staging' && (
          scanResults ? (
            <div className="rounded-lg border border-[#E5DDD6] bg-white p-6">
              <h3 className="font-semibold text-[#2C2825] mb-4">Severity staging</h3>
              <div className="space-y-2">
                {(scanResults.severityStages || []).map((s) => (
                  <div
                    key={s.level}
                    className={`rounded-lg border px-4 py-3 ${s.current ? 'bg-[#FCF3F0] border-[#E8C8BE]' : 'bg-white border-[#E5DDD6]'}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-semibold text-[#2C2825]">
                        Stage {s.level}: {s.label}
                      </div>
                      {s.current ? (
                        <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-[#A85D4A] text-white">
                          Current
                        </span>
                      ) : null}
                    </div>
                    <p className="text-xs text-[#736D68] mt-1">{s.description}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState title="Severity Staging" />
          )
        )}

        {activeTab === 'dca' && (
          <EmptyState title="Decision Curve" />
        )}
      </main>

      {/* Floating ChatBot Assistant */}
      {scanResults ? (
        <>
          {!isChatBotOpen ? (
            <button
              onClick={() => setIsChatBotOpen(true)}
              className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full bg-[#A85D4A] text-white shadow-lg hover:bg-[#935246] transition-colors print:hidden"
              title="Open ChatBot"
              aria-label="Open ChatBot"
            >
              <span className="sr-only">Open ChatBot</span>
              <Sparkles className="mx-auto h-5 w-5" />
            </button>
          ) : (
            <div className="fixed bottom-6 right-6 z-50 w-[360px] sm:w-[400px] rounded-2xl border border-[#E8C8BE] bg-white shadow-2xl overflow-hidden print:hidden">
              <div className="flex items-center justify-between px-4 py-3 bg-[#FCF3F0] border-b border-[#E8C8BE]">
                <div>
                  <h3 className="font-semibold text-[#2C2825]">ChatBot</h3>
                  <p className="text-[11px] text-[#736D68]">Scan assistant · {chatLoading ? 'typing…' : `scan #${scanResults.scan_id ?? '—'}`}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      void sendChat(
                        'Give me a concise summary of my scan results and next steps in simple language.'
                      );
                    }}
                    disabled={chatLoading || !scanResults?.scan_id}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-white border border-[#E5DDD6] text-[#A85D4A] hover:bg-[#F7F5F0] disabled:opacity-50 text-[11px] font-semibold"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    Summary
                  </button>
                  <button
                    onClick={() => setIsChatBotOpen(false)}
                    className="h-8 w-8 rounded-lg bg-white border border-[#E5DDD6] text-[#736D68] hover:bg-[#F7F5F0]"
                    aria-label="Close ChatBot"
                    title="Close"
                  >
                    ×
                  </button>
                </div>
              </div>

              <div className="p-3 h-72 overflow-auto bg-[#F7F5F0]">
                {messages.length === 0 ? (
                  <div className="text-sm text-[#736D68] leading-relaxed">
                    Ask me about your report. Try: <span className="font-semibold text-[#2C2825]">What does this diagnosis mean?</span>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {messages.map((m, i) => (
                      <div
                        key={i}
                        className={[
                          'max-w-[92%] rounded-xl px-3 py-2 text-sm leading-relaxed border shadow-sm',
                          m.role === 'user'
                            ? 'ml-auto bg-white border-[#E5DDD6] text-[#2C2825]'
                            : 'mr-auto bg-[#FCF3F0] border-[#E8C8BE] text-[#2C2825]',
                        ].join(' ')}
                      >
                        {m.content}
                      </div>
                    ))}
                    {chatLoading ? (
                      <div className="mr-auto max-w-[92%] rounded-xl px-3 py-2 text-sm border border-[#E8C8BE] bg-[#FCF3F0] text-[#736D68]">
                        typing…
                      </div>
                    ) : null}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              <div className="px-3 pt-2 pb-1 flex flex-wrap gap-2 border-t border-[#EEE7DF]">
                {[
                  'What does this diagnosis mean?',
                  'What should I do next?',
                  'Is this urgent?',
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => void sendChat(q)}
                    disabled={chatLoading}
                    className="px-3 py-1.5 rounded-full border border-[#E5DDD6] bg-white text-xs font-semibold text-[#A85D4A] hover:bg-[#F7F5F0] disabled:opacity-50 transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>

              <div className="p-3 flex items-end gap-2 border-t border-[#EEE7DF] bg-white">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask ChatBot…"
                  rows={2}
                  className="flex-1 resize-none px-3 py-2 rounded-lg border border-[#E5DDD6] bg-white text-sm text-[#2C2825] placeholder-[#A39A8E] focus:outline-none focus:ring-0 focus:border-[#A85D4A]/60"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      const msg = input;
                      setInput('');
                      void sendChat(msg);
                    }
                  }}
                  disabled={chatLoading}
                />
                <button
                  onClick={() => {
                    const msg = input;
                    setInput('');
                    void sendChat(msg);
                  }}
                  disabled={chatLoading}
                  className="px-4 py-2 rounded-lg bg-[#A85D4A] hover:bg-[#935246] disabled:opacity-50 text-white text-sm font-semibold transition-colors"
                >
                  Send
                </button>
                <button
                  onClick={() => setMessages([])}
                  disabled={chatLoading || messages.length === 0}
                  className="px-3 py-2 rounded-lg bg-white border border-[#E5DDD6] text-[#736D68] hover:bg-[#F7F5F0] disabled:opacity-50 text-sm"
                  title="Clear chat"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
