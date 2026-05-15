'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Eye, LogOut, Printer, Sparkles, Trash2, Upload } from 'lucide-react';

import { ClinicalFeaturesTab } from './dashboard/clinical-features-tab';
import { SeverityStagingTab } from './dashboard/severity-staging-tab';
import { DecisionCurveAnalysis } from './dashboard/decision-curve-analysis';
import { PrintReport } from './dashboard/print-report';
import { MOCK_RESULTS } from './dashboard/mock-data';
import { apiUrl } from '@/lib/api-base';

type ClinicalDashboardProps = {
  onLogout: () => void;
};

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
  aiConfidence: number;
  processingTime?: string;
  modelVersion?: string;
  recommendation: string;
  generatedReport?: string;
  clinical_report?: string;
  patient_report?: string;
  gradcam_base64?: string;
  clinicalFeatures: Array<{
    name: string;
    value: string;
    unit?: string;
    reference?: string;
    status: 'normal' | 'warning' | 'elevated' | 'critical';
    score: number;
    mask_key?: string;
  }>;
  segmentation_images?: {
    original_base64?: string;
    combined_overlay_base64?: string;
    masks_base64?: Record<string, string>;
    feature_images_base64?: Record<string, string>;
  };
  clinical_features?: {
    ma_count?: number;
    he_count?: number;
    ex_area_pixels?: number;
    se_area_pixels?: number;
    dme_risk_flag?: boolean;
  };
  classProbs: Record<string, number>;
  severityStages: Array<{
    level: number;
    label: string;
    description: string;
    current?: boolean;
  }>;
  dcaCurveData?: Array<{
    threshold: number;
    model: number;
    treatAll: number;
    treatNone: number;
  }>;
};

function SectionHeading({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-4 flex flex-col gap-1">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[#6E4B34]">
        {title}
      </div>
      {subtitle ? <p className="text-sm text-slate-500">{subtitle}</p> : null}
    </div>
  );
}

function EmptyState({ title, description }: { title: string; description: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
      <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-slate-50 border border-slate-200">
        <Eye className="h-6 w-6 text-[#A85D4A]" />
      </div>
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm text-slate-500">{description}</p>
    </div>
  );
}

function ScanningOverlay({ stageText }: { stageText: string }) {
  return (
    <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
      <div
        className="absolute inset-0"
        style={{
          background: `linear-gradient(to bottom,
            transparent 0%,
            rgba(168, 93, 74, 0.22) 48%,
            rgba(168, 93, 74, 0.55) 50%,
            rgba(168, 93, 74, 0.22) 52%,
            transparent 100%)`,
          animation: 'scan 2s ease-in-out infinite',
        }}
      />
      <div className="absolute inset-x-0 bottom-3 flex items-center justify-between px-3">
        <div className="rounded-full bg-white/90 px-3 py-1 text-[11px] font-semibold text-slate-800 border border-slate-200">
          {stageText}
        </div>
        <div className="inline-flex items-center gap-2 rounded-full bg-white/90 px-3 py-1 text-[11px] text-slate-500 border border-slate-200">
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-200 border-t-[#A85D4A]" />
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
  const inputRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<HTMLElement | null>(null);
  const resultsRef = useRef<HTMLElement | null>(null);
  const severityRef = useRef<HTMLElement | null>(null);
  const featuresRef = useRef<HTMLElement | null>(null);
  const dcaRef = useRef<HTMLElement | null>(null);
  const notesRef = useRef<HTMLElement | null>(null);
  const reportsRef = useRef<HTMLElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const [activeSection, setActiveSection] = useState<'scan-upload' | 'severity-staging' | 'clinical-features' | 'decision-curve' | 'notes' | 'generated-reports'>('scan-upload');
  const [scanResults, setScanResults] = useState<ScanResults | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reportType, setReportType] = useState<'clinical' | 'patient'>('clinical');
  const [showAiReport, setShowAiReport] = useState(false);
  const [isChatBotOpen, setIsChatBotOpen] = useState(false);
  const [previewMode, setPreviewMode] = useState<'original' | 'gradcam'>('original');
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'ai'; content: string }>>([]);
  const [input, setInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [authSession, setAuthSession] = useState<AuthSession | null>(null);
  const [processingStage, setProcessingStage] = useState(0);
  const [decisionCurveData, setDecisionCurveData] = useState<ScanResults['dcaCurveData']>(MOCK_RESULTS.dcaCurveData);

  const processingStages = useMemo(
    () => [
      'Applying fundus mask + CLAHE…',
      'Running EfficientNet‑B3…',
      'Applying clinical thresholds…',
      'Compiling clinical report…',
    ],
    []
  );

  const sections = useMemo(
    () => [
      { id: 'scan-upload', label: 'Scan Upload', ref: uploadRef },
      { id: 'severity-staging', label: 'Severity Staging', ref: severityRef },
      { id: 'clinical-features', label: 'Clinical Features', ref: featuresRef },
      { id: 'decision-curve', label: 'Decision Curve Analysis', ref: dcaRef },
      { id: 'notes', label: 'Notes', ref: notesRef },
      { id: 'generated-reports', label: 'Reports', ref: reportsRef },
    ],
    []
  );

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
    const timer = setInterval(() => {
      setProcessingStage((previous) => Math.min(previous + 1, processingStages.length - 1));
    }, 850);
    return () => clearInterval(timer);
  }, [isProcessing, processingStages.length]);

  useEffect(() => {
    const elements = sections
      .map((section) => section.ref.current)
      .filter((element): element is HTMLElement => Boolean(element));

    if (!elements.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (!visible) return;
        const found = sections.find((section) => section.ref.current === visible.target);
        if (found) {
          setActiveSection(found.id as typeof activeSection);
        }
      },
      {
        root: null,
        threshold: [0.2, 0.35, 0.5, 0.7],
        rootMargin: '-20% 0px -55% 0px',
      }
    );

    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [sections, activeSection]);

  useEffect(() => {
    if (scanResults) {
      setShowAiReport(true);
    }
  }, [scanResults]);

  useEffect(() => {
    let active = true;

    async function loadDecisionCurve() {
      if (!scanResults) {
        setDecisionCurveData(MOCK_RESULTS.dcaCurveData);
        return;
      }

      try {
        const response = await fetch(apiUrl('/api/decision-curve'));
        const payload = await response.json();
        const curve = Array.isArray(payload) ? payload : payload?.data;
        if (!response.ok || !Array.isArray(curve) || curve.length === 0) {
          throw new Error(payload?.detail || 'Decision curve data unavailable');
        }
        if (active) {
          setDecisionCurveData(curve);
        }
      } catch {
        if (active) {
          setDecisionCurveData(
            scanResults?.dcaCurveData && scanResults.dcaCurveData.length > 0
              ? scanResults.dcaCurveData
              : MOCK_RESULTS.dcaCurveData
          );
        }
      }
    }

    void loadDecisionCurve();

    return () => {
      active = false;
    };
  }, [scanResults]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

  const handlePrint = () => {
    window.print();
  };

  const handleImageUpload = async (file: File) => {
    setError(null);
    setScanResults(null);
    setPreviewMode('original');

    const previewUrl = URL.createObjectURL(file);
    setSelectedImage(previewUrl);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(apiUrl('/api/analyze'), {
        method: 'POST',
        headers: authSession?.user?.email ? { 'X-User-Email': authSession.user.email } : undefined,
        body: formData,
      });

      const json = (await response.json()) as any;
      if (!response.ok || json?.status !== 'success') {
        throw new Error(json?.message || 'Analysis failed');
      }

      const data = json?.data;
      if (!data?.diagnosis) {
        throw new Error('Malformed response from server (missing diagnosis).');
      }

      setScanResults(data as ScanResults);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (exception: any) {
      setError(exception?.message || 'Upload failed');
      uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } finally {
      setIsProcessing(false);
    }
  };

  const severityScore = useMemo(() => {
    if (!scanResults) return 0;
    const denominator = Number(scanResults.maxSeverity || 5);
    const numerator = Number(scanResults.severity || 0);
    if (!denominator) return 0;
    return Math.max(0, Math.min(100, (numerator / denominator) * 100));
  }, [scanResults]);

  const narrativeParagraphs = useMemo(() => {
    const chosen =
      reportType === 'clinical'
        ? scanResults?.clinical_report || scanResults?.generatedReport || ''
        : scanResults?.patient_report || '';
    return (chosen || '')
      .trim()
      .split(/\n+/)
      .map((paragraph) => paragraph.trim())
      .filter(Boolean);
  }, [scanResults, reportType]);

  const fallbackNarrativeParagraphs = useMemo(() => {
    if (!scanResults) return [];
    const clinicalFeaturesTop = (scanResults.clinicalFeatures || [])
      .slice(0, 3)
      .map((feature) => `${feature.name}: ${feature.value}${feature.unit ? ` ${feature.unit}` : ''}`)
      .join(' • ');

    if (reportType === 'clinical') {
      return [
        `Diagnosis: ${scanResults.diagnosis} (${scanResults.etdrsLevel}). ICD-10: ${scanResults.icd10}.`,
        `Model confidence is ${scanResults.aiConfidence.toFixed(1)}% with severity score ${severityScore.toFixed(0)}%.`,
        clinicalFeaturesTop
          ? `Key extracted features: ${clinicalFeaturesTop}.`
          : 'Feature extraction was completed; detailed metrics are available in Clinical Features.',
        `Recommendation: ${scanResults.recommendation}`,
      ];
    }

    return [
      `Your eye scan suggests: ${scanResults.diagnosis}.`,
      `Confidence is ${scanResults.aiConfidence.toFixed(1)}%. This supports the doctor’s decision, but it is not the final diagnosis by itself.`,
      `Recommended next step: ${scanResults.recommendation}`,
    ];
  }, [scanResults, reportType, severityScore]);

  const gradcamSrc = useMemo(() => {
    const base64 = scanResults?.gradcam_base64;
    return base64 ? `data:image/png;base64,${base64}` : null;
  }, [scanResults]);

  const reportPatient = useMemo(() => {
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');

    const resolvedDob = scanResults?.patient_dob || authSession?.patient_dob || '';
    let resolvedAge: number | undefined;
    if (resolvedDob) {
      const dobDate = new Date(resolvedDob);
      if (!Number.isNaN(dobDate.getTime())) {
        let age = today.getFullYear() - dobDate.getFullYear();
        const birthdayPassed =
          today.getMonth() > dobDate.getMonth() ||
          (today.getMonth() === dobDate.getMonth() && today.getDate() >= dobDate.getDate());
        if (!birthdayPassed) age -= 1;
        resolvedAge = Math.max(age, 0);
      }
    }

    return {
      id: scanResults?.patient_uid || authSession?.patient_uid || '—',
      name: scanResults?.patient_name || authSession?.patient_name || '—',
      dob: resolvedDob || '—',
      age: resolvedAge ?? '—',
      eye: 'Right Eye (OD)',
      examDate: `${year}-${month}-${day}`,
      referringPhysician:
        scanResults?.doctor_name || authSession?.doctor_name || authSession?.user?.email || '—',
    };
  }, [scanResults, authSession]);

  const scrollToSection = (sectionId: typeof activeSection) => {
    const section = sections.find((entry) => entry.id === sectionId);
    section?.ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleChatSend = async (text: string) => {
    const message = text.trim();
    if (!message) return;

    const scanId = scanResults?.scan_id;
    if (!scanId) {
      setMessages((previous) => [
        ...previous,
        { role: 'ai', content: 'Please analyze a scan first so I can answer with your specific context.' },
      ]);
      return;
    }

    setMessages((previous) => [...previous, { role: 'user', content: message }]);
    setChatLoading(true);

    try {
      const response = await fetch(apiUrl('/api/chat'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_id: scanId, message }),
      });
      const json = (await response.json()) as any;
      if (!response.ok || !json?.reply) {
        throw new Error(json?.detail || json?.message || 'Chat request failed');
      }
      setMessages((previous) => [...previous, { role: 'ai', content: String(json.reply) }]);
    } catch (exception: any) {
      setMessages((previous) => [
        ...previous,
        { role: 'ai', content: exception?.message || 'Unable to reach the chat service right now.' },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#FCF8F3] text-[#3B2416] scroll-smooth">
      <PrintReport
        patient={reportPatient as any}
        results={scanResults as any}
        originalImageSrc={selectedImage}
        gradcamImageSrc={gradcamSrc}
      />

      <header className="sticky top-0 z-50 border-b border-[#E8D9C8] bg-[#FCF8F3]/95 backdrop-blur print:hidden">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#3B2416]">
                <Eye className="h-5 w-5 text-white" />
              </div>
              <div>
                <p className="text-lg font-bold text-[#3B2416] leading-none">Opthalmic</p>
                <p className="text-xs text-[#6E4B34] mt-1">Clinical Diagnosis</p>
              </div>
              <div className="hidden sm:block border-l border-[#E8D9C8] pl-4">
                <p className="text-xs text-[#6E4B34]">Patient ID</p>
                <p className="font-semibold text-[#3B2416]">{scanResults?.patient_uid || authSession?.patient_uid || '—'}</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                onClick={handlePrint}
                disabled={!scanResults}
                className="inline-flex items-center gap-2 rounded-lg bg-[#3B2416] px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-[#2F1B11] disabled:opacity-50"
              >
                <Printer className="h-4 w-4" />
                Export PDF
              </button>
              <button
                onClick={() => {
                  try {
                    localStorage.removeItem('opthalmic_session');
                  } catch {}
                  onLogout();
                }}
                className="inline-flex items-center gap-2 rounded-lg border border-[#D6C3AE] px-4 py-2 text-sm font-medium text-[#3B2416] transition-colors hover:bg-[#F4EADD]"
              >
                <LogOut className="h-4 w-4" />
                Logout
              </button>
            </div>
          </div>

          <nav className="mt-5 overflow-x-auto border-b border-[#E8D9C8] pb-3">
            {sections.map((section) => {
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  onClick={() => scrollToSection(section.id as typeof activeSection)}
                  className={`mr-6 border-b-2 pb-1 text-sm font-medium transition-colors ${
                    isActive
                      ? 'border-[#3B2416] text-[#3B2416]'
                      : 'border-transparent text-[#6E4B34] hover:border-[#D6C3AE] hover:text-[#3B2416]'
                  }`}
                >
                  {section.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8 space-y-8">
        <section ref={uploadRef} id="scan-upload" className="scroll-mt-28">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <SectionHeading title="Scan Upload" subtitle="Upload a retinal fundus image to start the AI-assisted grading workflow." />

            <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-5">
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleImageUpload(file);
                }}
              />

              <div
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  const file = event.dataTransfer.files?.[0];
                  if (file) void handleImageUpload(file);
                }}
                className="relative cursor-pointer rounded-2xl border-2 border-dashed border-slate-300 bg-white p-5 transition-colors hover:bg-slate-50"
              >
                <div className="min-h-[280px] grid place-items-center text-center">
                  {selectedImage ? (
                    <div className="w-full">
                      <div className="relative overflow-hidden rounded-2xl border border-slate-200 bg-white">
                        <img src={selectedImage} alt="Preview" className="h-72 w-full object-contain" />
                        {isProcessing ? <ScanningOverlay stageText={processingStages[processingStage]} /> : null}
                      </div>
                      <div className="mt-3 text-xs text-slate-500">
                        {isProcessing ? 'Processing… please keep this tab open.' : 'Click again to upload a different scan.'}
                      </div>
                    </div>
                  ) : (
                    <div className="max-w-sm">
                      <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#FCF3F0] text-[#A85D4A]">
                        <Upload className="h-6 w-6" />
                      </div>
                      <p className="text-base font-semibold text-slate-900">Drag & drop or click to browse</p>
                      <p className="mt-1 text-sm text-slate-500">Upload a retinal fundus scan to generate clinical insights.</p>
                      <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-500">
                        {['PNG', 'JPG', 'JPEG', 'TIFF'].map((format) => (
                          <span key={format} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1">
                            {format}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {error ? <div className="mt-5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</div> : null}
            </div>
          </div>
        </section>

        {scanResults ? (
          <>
            <section ref={resultsRef} className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading
                  title="Analysis Summary"
                  subtitle="Use the toggle to switch between the original scan and the Grad-CAM heatmap."
                />

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1">
                    <button
                      onClick={() => setPreviewMode('original')}
                      className={[
                        'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                        previewMode === 'original'
                          ? 'bg-[#A85D4A] text-white'
                          : 'text-slate-500 hover:text-slate-900',
                      ].join(' ')}
                    >
                      Original Image
                    </button>
                    <button
                      onClick={() => setPreviewMode('gradcam')}
                      disabled={!gradcamSrc}
                      className={[
                        'px-3 py-1.5 rounded-md text-xs font-semibold transition-colors',
                        previewMode === 'gradcam'
                          ? 'bg-[#A85D4A] text-white'
                          : 'text-slate-500 hover:text-slate-900',
                        !gradcamSrc ? 'cursor-not-allowed opacity-40 hover:text-slate-500' : '',
                      ].join(' ')}
                    >
                      Grad-CAM
                    </button>
                  </div>

                  <div className="mt-3 flex h-[420px] items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white">
                    {previewMode === 'gradcam' ? (
                      gradcamSrc ? (
                        <img
                          src={gradcamSrc}
                          alt="Grad-CAM heatmap"
                          className="max-h-full max-w-full object-contain"
                        />
                      ) : (
                        <div className="grid h-full w-full place-items-center px-6 text-center text-sm text-slate-500">
                          Grad-CAM is not available for this scan.
                        </div>
                      )
                    ) : (
                      <img
                        src={selectedImage ?? ''}
                        alt="Original retinal scan"
                        className="max-h-full max-w-full object-contain"
                      />
                    )}
                  </div>
                </div>

                <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full bg-[#A85D4A]/10 px-3 py-1 text-xs font-semibold text-[#A85D4A]">
                      {scanResults.diagnosis}
                    </span>
                    <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600">
                      {scanResults.etdrsLevel}
                    </span>
                    <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs text-slate-600">
                      Confidence {scanResults.aiConfidence.toFixed(1)}%
                    </span>
                  </div>
                  <p className="mt-3 text-sm leading-7 text-slate-600">{scanResults.recommendation}</p>
                </div>
              </div>
            </section>

            <section className="scroll-mt-28">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
                {[
                  { label: 'Diagnosis', value: scanResults.diagnosis },
                  { label: 'ETDRS Level', value: scanResults.etdrsLevel },
                  { label: 'AI Confidence', value: `${scanResults.aiConfidence.toFixed(1)}%` },
                  { label: 'Severity Score', value: `${severityScore.toFixed(0)}%` },
                ].map((item) => (
                  <div key={item.label} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                    <p className="text-xs uppercase tracking-wide text-slate-500">{item.label}</p>
                    <p className="mt-1 text-base font-semibold text-slate-900">{item.value}</p>
                  </div>
                ))}
              </div>
            </section>

            <section ref={severityRef} id="severity-staging" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading title="Severity Staging" subtitle="ETDRS progression scale and current stage selection." />
                <SeverityStagingTab stages={(scanResults as any).severityStages} currentLevel={(scanResults as any).severity} />
              </div>
            </section>

            <section ref={featuresRef} id="clinical-features" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading title="Clinical Features" subtitle="Key lesion-level findings extracted from real-time segmentation masks." />
                <ClinicalFeaturesTab
                  features={(scanResults as any).clinicalFeatures}
                  segmentationImages={(scanResults as any).segmentation_images}
                />
              </div>
            </section>

            <section ref={dcaRef} id="decision-curve" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading title="Decision Curve Analysis" subtitle="Net clinical benefit across threshold probabilities and treatment strategies." />
                <DecisionCurveAnalysis
                  data={
                    decisionCurveData && decisionCurveData.length > 0
                      ? decisionCurveData
                      : (scanResults as any).dcaCurveData && (scanResults as any).dcaCurveData.length > 0
                        ? (scanResults as any).dcaCurveData
                        : MOCK_RESULTS.dcaCurveData
                  }
                />
              </div>
            </section>

            <section ref={notesRef} id="notes" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading title="Notes" subtitle="Per-scan technical context and interpretation guidance." />

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Terms</h3>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      <li><span className="font-medium text-slate-900">NPDR</span>: Non-Proliferative Diabetic Retinopathy.</li>
                      <li><span className="font-medium text-slate-900">PDR</span>: Proliferative Diabetic Retinopathy.</li>
                      <li><span className="font-medium text-slate-900">DME</span>: Diabetic Macular Edema risk flag from exudate proximity analysis.</li>
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Segmentation</h3>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      <li>Models generate binary masks for <span className="font-medium text-slate-900">OD, MA, HE, EX, SE</span>.</li>
                      <li>Connected-component cleanup removes tiny artifacts before counting area/components.</li>
                      <li>Overlay, lesion masks, and feature maps are included in current scan output and PDF.</li>
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Metrics</h3>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      <li>MA count: <span className="font-medium text-slate-900">{scanResults?.clinical_features?.ma_count ?? '—'}</span></li>
                      <li>HE count: <span className="font-medium text-slate-900">{scanResults?.clinical_features?.he_count ?? '—'}</span></li>
                      <li>EX area: <span className="font-medium text-slate-900">{scanResults?.clinical_features?.ex_area_pixels ?? '—'}</span> px</li>
                      <li>SE area: <span className="font-medium text-slate-900">{scanResults?.clinical_features?.se_area_pixels ?? '—'}</span> px</li>
                      <li>Severity score: <span className="font-medium text-slate-900">{severityScore.toFixed(0)}%</span></li>
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Risk</h3>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      <li><span className="font-medium text-slate-900">PDR conversion risk</span> is inferred from stage and lesion burden trend.</li>
                      <li><span className="font-medium text-slate-900">Vision loss risk</span> increases with advanced stage and edema-linked lesions.</li>
                      <li>DME risk (current): <span className="font-medium text-slate-900">{scanResults?.clinical_features?.dme_risk_flag ? 'High' : 'Low'}</span></li>
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 lg:col-span-2">
                    <h3 className="text-sm font-semibold text-slate-900">DCA</h3>
                    <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
                      <li><span className="font-medium text-slate-900">Model</span>: net benefit of model-guided treatment across thresholds.</li>
                      <li><span className="font-medium text-slate-900">Treat All</span>: hypothetical strategy of treating everyone.</li>
                      <li><span className="font-medium text-slate-900">Treat None</span>: baseline strategy with zero interventions.</li>
                      <li>Useful range is where <span className="font-medium text-slate-900">Model</span> stays above both comparator curves.</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>

            <section ref={reportsRef} id="generated-reports" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading title="Reports" subtitle="Clinical and patient-facing narrative summaries." />

                <div className="flex flex-wrap gap-2 border-b border-slate-200 pb-4">
                  {[
                    { id: 'clinical', label: 'Clinical View' },
                    { id: 'patient', label: 'Patient View' },
                  ].map((tab) => {
                    const isActive = reportType === tab.id;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setReportType(tab.id as 'clinical' | 'patient')}
                        className={`rounded-full px-4 py-2 text-sm font-medium transition-colors ${
                          isActive
                            ? 'bg-[#A85D4A] text-white'
                            : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
                        }`}
                      >
                        {tab.label}
                      </button>
                    );
                  })}
                </div>

                <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Clinical Narrative</h3>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-slate-600">
                      {(narrativeParagraphs.length > 0 ? narrativeParagraphs : fallbackNarrativeParagraphs).map((paragraph, index) => (
                        <p key={index}>{paragraph}</p>
                      ))}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="text-sm font-semibold text-slate-900">Quick Summary</h3>
                    <div className="mt-3 space-y-3 text-sm leading-7 text-slate-600">
                      <p>
                        Diagnosis: <span className="font-semibold text-slate-900">{scanResults.diagnosis}</span>
                      </p>
                      <p>
                        Severity: <span className="font-semibold text-slate-900">{scanResults.etdrsLevel}</span>
                      </p>
                      <p>
                        AI confidence: <span className="font-semibold text-slate-900">{scanResults.aiConfidence.toFixed(1)}%</span>
                      </p>
                      <p>{scanResults.recommendation}</p>
                    </div>
                  </div>
                </div>

              </div>
            </section>
          </>
        ) : (
          <>
            <section ref={severityRef} id="severity-staging" className="scroll-mt-28">
              <EmptyState title="Severity Staging" description="This section becomes available after a scan is analyzed." />
            </section>
            <section ref={featuresRef} id="clinical-features" className="scroll-mt-28">
              <EmptyState title="Clinical Features" description="This section becomes available after a scan is analyzed." />
            </section>
            <section ref={dcaRef} id="decision-curve" className="scroll-mt-28">
              <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <SectionHeading
                  title="Decision Curve Analysis"
                  subtitle="Preview curve is shown below. It will be replaced with patient-specific or cohort data after scan analysis."
                />
                <DecisionCurveAnalysis data={decisionCurveData || MOCK_RESULTS.dcaCurveData} />
              </div>
            </section>
            <section ref={notesRef} id="notes" className="scroll-mt-28">
              <EmptyState title="Notes" description="This section becomes available after a scan is analyzed." />
            </section>
            <section ref={reportsRef} id="generated-reports" className="scroll-mt-28">
              <EmptyState title="Reports" description="This section becomes available after a scan is analyzed." />
            </section>
          </>
        )}
      </main>

      {scanResults ? (
        <>
          {!isChatBotOpen ? (
            <button
              onClick={() => setIsChatBotOpen(true)}
              className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full bg-[#A85D4A] text-white shadow-lg transition-colors hover:bg-[#935246] print:hidden"
              title="Open ChatBot"
              aria-label="Open ChatBot"
            >
              <span className="sr-only">Open ChatBot</span>
              <Sparkles className="mx-auto h-5 w-5" />
            </button>
          ) : (
            <div className="fixed bottom-6 right-6 z-50 w-[360px] overflow-hidden rounded-2xl border border-[#E8C8BE] bg-white shadow-2xl print:hidden sm:w-[400px]">
              <div className="flex items-center justify-between border-b border-[#E8C8BE] bg-[#FCF3F0] px-4 py-3">
                <div>
                  <h3 className="font-semibold text-[#2C2825]">ChatBot</h3>
                  <p className="text-[11px] text-[#736D68]">Scan assistant · {chatLoading ? 'typing…' : `scan #${scanResults.scan_id ?? '—'}`}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      void handleChatSend('Give me a concise summary of my scan results and next steps in simple language.');
                    }}
                    disabled={chatLoading || !scanResults?.scan_id}
                    className="inline-flex items-center gap-1 rounded-lg border border-[#E5DDD6] bg-white px-2.5 py-1.5 text-[11px] font-semibold text-[#A85D4A] transition-colors hover:bg-[#F7F5F0] disabled:opacity-50"
                  >
                    <Sparkles className="h-3.5 w-3.5" />
                    Summary
                  </button>
                  <button
                    onClick={() => setIsChatBotOpen(false)}
                    className="h-8 w-8 rounded-lg border border-[#E5DDD6] bg-white text-[#736D68] hover:bg-[#F7F5F0]"
                    aria-label="Close ChatBot"
                    title="Close"
                  >
                    ×
                  </button>
                </div>
              </div>

              <div className="h-72 overflow-auto bg-[#F7F5F0] p-3">
                {messages.length === 0 ? (
                  <div className="text-sm leading-relaxed text-[#736D68]">
                    Ask me about your report. Try: <span className="font-semibold text-[#2C2825]">What does this diagnosis mean?</span>
                  </div>
                ) : (
                  <div className="flex flex-col gap-2">
                    {messages.map((message, index) => (
                      <div
                        key={index}
                        className={[
                          'max-w-[92%] rounded-xl border px-3 py-2 text-sm leading-relaxed shadow-sm',
                          message.role === 'user'
                            ? 'ml-auto border-[#E5DDD6] bg-white text-[#2C2825]'
                            : 'mr-auto border-[#E8C8BE] bg-[#FCF3F0] text-[#2C2825]',
                        ].join(' ')}
                      >
                        {message.content}
                      </div>
                    ))}
                    {chatLoading ? (
                      <div className="mr-auto max-w-[92%] rounded-xl border border-[#E8C8BE] bg-[#FCF3F0] px-3 py-2 text-sm text-[#736D68]">
                        typing…
                      </div>
                    ) : null}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              <div className="flex flex-wrap gap-2 border-t border-[#EEE7DF] px-3 pt-2 pb-1">
                {['What does this diagnosis mean?', 'What should I do next?', 'Is this urgent?'].map((question) => (
                  <button
                    key={question}
                    onClick={() => void handleChatSend(question)}
                    disabled={chatLoading}
                    className="rounded-full border border-[#E5DDD6] bg-white px-3 py-1.5 text-xs font-semibold text-[#A85D4A] transition-colors hover:bg-[#F7F5F0] disabled:opacity-50"
                  >
                    {question}
                  </button>
                ))}
              </div>

              <div className="flex items-end gap-2 border-t border-[#EEE7DF] bg-white p-3">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Ask Copilot…"
                  rows={2}
                  className="flex-1 resize-none rounded-lg border border-[#E5DDD6] bg-white px-3 py-2 text-sm text-[#2C2825] placeholder-[#A39A8E] focus:border-[#A85D4A]/60 focus:outline-none focus:ring-0"
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault();
                      const message = input;
                      setInput('');
                      void handleChatSend(message);
                    }
                  }}
                  disabled={chatLoading}
                />
                <button
                  onClick={() => {
                    const message = input;
                    setInput('');
                    void handleChatSend(message);
                  }}
                  disabled={chatLoading}
                  className="rounded-lg bg-[#A85D4A] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#935246] disabled:opacity-50"
                >
                  Send
                </button>
                <button
                  onClick={() => setMessages([])}
                  disabled={chatLoading || messages.length === 0}
                  className="rounded-lg border border-[#E5DDD6] bg-white px-3 py-2 text-sm text-[#736D68] transition-colors hover:bg-[#F7F5F0] disabled:opacity-50"
                  title="Clear chat"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
