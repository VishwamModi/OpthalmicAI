'use client';

import { CheckCircle2, AlertCircle } from 'lucide-react';
import { MOCK_RESULTS } from './mock-data';

interface SeverityStagingTabProps {
  stages: typeof MOCK_RESULTS['severityStages'];
  currentLevel: typeof MOCK_RESULTS['severity'];
}

const SEVERITY_COLORS = ['#6B705C', '#A39A8E', '#A85D4A', '#CB997E', '#9E2A2B'];
const SEVERITY_BG = ['#F0F4EE', '#F2EFE9', '#FCF3F0', '#FDF8F4', '#FCF0EF'];

export function SeverityStagingTab({ stages, currentLevel }: SeverityStagingTabProps) {
  return (
    <div className="flex flex-col gap-6">
      {/* Timeline */}
      <div className="rounded-lg border border-[#E5DDD6] bg-white p-6">
        <h3 className="font-semibold text-[#2C2825] mb-6">Diabetic Retinopathy Severity Scale (ETDRS)</h3>
        <div className="relative">
          {/* Connecting line */}
          <div className="absolute left-[22px] top-6 bottom-6 w-0.5 bg-[#E5DDD6]" />

          <div className="flex flex-col gap-5">
            {stages.map((stage) => {
              const isCurrent = stage.level === currentLevel;
              const isPast = stage.level < currentLevel;
              const color = SEVERITY_COLORS[stage.level - 1];
              const bg = SEVERITY_BG[stage.level - 1];

              return (
                <div key={stage.level} className="flex items-start gap-4 relative">
                  {/* Circle */}
                  <div
                    className={`relative z-10 w-11 h-11 rounded-full flex items-center justify-center shrink-0 transition-all border-2 ${
                      isCurrent ? "shadow-lg scale-110" : ""
                    }`}
                    style={{
                      background: isCurrent || isPast ? bg : '#F7F5F0',
                      borderColor: isCurrent || isPast ? color : '#E5DDD6',
                    }}
                  >
                    {isCurrent ? (
                      <AlertCircle className="w-5 h-5" style={{ color }} />
                    ) : isPast ? (
                      <CheckCircle2 className="w-5 h-5" style={{ color }} />
                    ) : (
                      <span className="text-sm font-bold text-[#736D68]">{stage.level}</span>
                    )}
                  </div>

                  {/* Content */}
                  <div
                    className={`flex-1 rounded-lg p-4 border transition-all ${
                      isCurrent ? 'shadow-sm' : 'bg-transparent'
                    }`}
                    style={{
                      background: isCurrent ? bg : 'transparent',
                      borderColor: isCurrent ? color : 'transparent',
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="text-sm font-bold"
                        style={{ color: isCurrent || isPast ? color : '#736D68' }}
                      >
                        Stage {stage.level}: {stage.label}
                      </span>
                      {isCurrent && (
                        <span
                          className="px-2 py-0.5 text-xs font-semibold rounded-full text-white"
                          style={{ background: color }}
                        >
                          Current
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[#736D68] mt-1 leading-relaxed">{stage.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Progression risk */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            label: '1-Year Progression Risk',
            value: '~20%',
            sub: 'to Severe NPDR',
            color: '#A85D4A',
            bg: '#FCF3F0',
            border: '#E8C8BE',
          },
          {
            label: 'PDR Conversion Risk',
            value: '~5%',
            sub: 'within 2 years',
            color: '#CB997E',
            bg: '#FDF8F4',
            border: '#ECC9B0',
          },
          {
            label: 'Vision Loss Risk',
            value: 'Low',
            sub: 'with proper management',
            color: '#6B705C',
            bg: '#F0F4EE',
            border: '#D4CCC2',
          },
        ].map((item) => (
          <div
            key={item.label}
            className="rounded-lg border p-4 flex flex-col gap-1"
            style={{ background: item.bg, borderColor: item.border }}
          >
            <p className="text-xs text-[#736D68] font-medium">{item.label}</p>
            <p className="text-2xl font-bold" style={{ color: item.color }}>{item.value}</p>
            <p className="text-xs text-[#736D68]">{item.sub}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
