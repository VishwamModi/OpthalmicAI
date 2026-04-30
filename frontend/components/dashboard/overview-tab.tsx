'use client';

import { Activity, AlertTriangle, CheckCircle2, XCircle, Clock, Cpu } from 'lucide-react';
import { MOCK_PATIENT, MOCK_RESULTS } from './mock-data';

interface OverviewTabProps {
  patient: typeof MOCK_PATIENT;
  results: typeof MOCK_RESULTS;
}

const STATUS_CONFIG = {
  normal: { label: 'Normal', color: '#6B705C', bg: '#F0F4EE', border: '#D4CCC2' },
  warning: { label: 'Borderline', color: '#CB997E', bg: '#FDF8F4', border: '#ECC9B0' },
  elevated: { label: 'Elevated', color: '#A85D4A', bg: '#FCF3F0', border: '#E8C8BE' },
  critical: { label: 'Critical', color: '#9E2A2B', bg: '#FCF0EF', border: '#E8B5B6' },
};

const SEVERITY_COLORS = ['#6B705C', '#A39A8E', '#A85D4A', '#CB997E', '#9E2A2B'];

function StatusBadge({ status }: { status: keyof typeof STATUS_CONFIG }) {
  const cfg = STATUS_CONFIG[status];
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border"
      style={{ color: cfg.color, background: cfg.bg, borderColor: cfg.border }}
    >
      {status === "normal" ? (
        <CheckCircle2 className="w-3 h-3" />
      ) : status === "warning" ? (
        <AlertTriangle className="w-3 h-3" />
      ) : (
        <XCircle className="w-3 h-3" />
      )}
      {cfg.label}
    </span>
  );
}

export function OverviewTab({ patient, results }: OverviewTabProps) {
  const r = results;
  const severityColor = SEVERITY_COLORS[r.severity - 1];
  const confidenceEntries = r.classProbs
    ? Object.entries(r.classProbs as Record<string, number>).map(([k, v]) => ({ label: k, p: Number(v) }))
    : [];

  return (
    <div className="flex flex-col gap-6">
      {confidenceEntries.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-[#E5DDD6] bg-white p-5">
            <h3 className="font-semibold text-[#2C2825] mb-4">Model Confidence</h3>
            <div className="flex flex-col gap-3">
              {confidenceEntries.length === 0 ? (
                <p className="text-sm text-[#736D68]">No confidence scores available.</p>
              ) : (
                confidenceEntries
                  .sort((a, b) => b.p - a.p)
                  .map((row) => (
                    <div key={row.label} className="flex flex-col gap-1.5">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-[#2C2825]">{row.label}</span>
                        <span className="font-mono text-[#736D68]">{(row.p * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 rounded-full bg-[#E5DDD6] overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.max(0, Math.min(100, row.p * 100))}%`, background: '#A85D4A' }}
                        />
                      </div>
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* Primary result banner */}
      <div className="rounded-lg border border-[#E5DDD6] bg-white p-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-sm text-[#736D68] font-medium">
            <Activity className="w-4 h-4 text-[#A85D4A]" />
            Primary Diagnosis
          </div>
          <h2 className="text-2xl font-bold text-[#2C2825]">{r.diagnosis}</h2>
          <div className="flex flex-wrap gap-2 mt-1">
            <span className="px-2.5 py-1 rounded-lg bg-[#A85D4A]/10 text-[#A85D4A] text-xs font-semibold border border-[#A85D4A]/20">
              {r.etdrsLevel}
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-[#F7F5F0] text-[#736D68] text-xs font-medium border border-[#E5DDD6]">
              ICD-10: {r.icd10}
            </span>
          </div>
        </div>

        {/* AI Confidence */}
        <div className="flex flex-col items-center gap-1 min-w-[120px]">
          <div className="relative w-24 h-24">
            <svg viewBox="0 0 36 36" className="w-24 h-24 -rotate-90">
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="#E5DDD6" strokeWidth="2.8" />
              <circle
                cx="18" cy="18" r="15.9" fill="none"
                stroke="#A85D4A" strokeWidth="2.8"
                strokeDasharray={`${r.aiConfidence} ${100 - r.aiConfidence}`}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-xl font-bold text-[#2C2825]">{r.aiConfidence}%</span>
            </div>
          </div>
          <span className="text-xs text-[#736D68] font-medium">AI Confidence</span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: Activity, label: 'Severity Level', value: `${r.severity} / ${r.maxSeverity}`, color: severityColor },
          { icon: Clock, label: 'Processing Time', value: r.processingTime, color: '#A85D4A' },
          { icon: Cpu, label: 'Model Version', value: r.modelVersion, color: '#A39A8E' },
          {
            icon: AlertTriangle,
            label: 'Features Flagged',
            value: `${r.clinicalFeatures.filter((f) => f.status !== 'normal').length} / ${r.clinicalFeatures.length}`,
            color: '#CB997E',
          },
        ].map((stat) => (
          <div key={stat.label} className="rounded-lg border border-[#E5DDD6] bg-white p-4 flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs text-[#736D68] font-medium">
              <stat.icon className="w-3.5 h-3.5" style={{ color: stat.color }} />
              {stat.label}
            </div>
            <p className="text-lg font-bold text-[#2C2825]">{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Feature summary table */}
      <div className="rounded-lg border border-[#E5DDD6] bg-white overflow-hidden">
        <div className="px-6 py-4 border-b border-[#E5DDD6]">
          <h3 className="font-semibold text-[#2C2825]">Clinical Feature Summary</h3>
          <p className="text-xs text-[#736D68] mt-0.5">Top findings from AI segmentation analysis</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-[#F7F5F0]">
                <th className="text-left px-6 py-3 text-xs font-semibold text-[#736D68] uppercase tracking-wide">Feature</th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-[#736D68] uppercase tracking-wide">Detected Value</th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-[#736D68] uppercase tracking-wide">AI Score</th>
                <th className="text-left px-6 py-3 text-xs font-semibold text-[#736D68] uppercase tracking-wide">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5DDD6]">
              {r.clinicalFeatures.map((f) => (
                <tr key={f.name} className="hover:bg-[#F7F5F0] transition-colors">
                  <td className="px-6 py-3.5 font-medium text-[#2C2825]">{f.name}</td>
                  <td className="px-6 py-3.5 text-[#736D68]">{f.value}</td>
                  <td className="px-6 py-3.5">
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-1.5 rounded-full bg-[#E5DDD6] max-w-[80px]">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${f.score * 100}%`,
                            background: STATUS_CONFIG[f.status as keyof typeof STATUS_CONFIG]?.color ?? '#736D68',
                          }}
                        />
                      </div>
                      <span className="text-xs text-[#736D68] w-8">{(f.score * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-3.5">
                    <StatusBadge status={f.status as keyof typeof STATUS_CONFIG} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recommendation */}
      <div className="rounded-lg border border-[#A85D4A]/20 bg-[#A85D4A]/5 p-5">
        <h3 className="font-semibold text-[#2C2825] mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4 text-[#A85D4A]" />
          Clinical Recommendation
        </h3>
        <p className="text-sm text-[#736D68] leading-relaxed">{r.recommendation}</p>
      </div>
    </div>
  );
}
