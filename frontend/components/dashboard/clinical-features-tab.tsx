'use client';

import { CheckCircle2, AlertTriangle, XCircle, Expand } from 'lucide-react';
import { useState } from 'react';
import { MOCK_RESULTS } from './mock-data';

type SegmentationImages = {
  feature_images_base64?: Record<string, string>;
  masks_base64?: Record<string, string>;
  combined_overlay_base64?: string;
  original_base64?: string;
};

interface ClinicalFeaturesTaProps {
  features: Array<(typeof MOCK_RESULTS['clinicalFeatures'][number]) & { mask_key?: string }>;
  segmentationImages?: SegmentationImages | null;
}

const STATUS_CONFIG = {
  normal: { label: 'Normal', color: '#6B705C', bg: '#F0F4EE', border: '#D4CCC2', Icon: CheckCircle2 },
  warning: { label: 'Borderline', color: '#CB997E', bg: '#FDF8F4', border: '#ECC9B0', Icon: AlertTriangle },
  elevated: { label: 'Elevated', color: '#A85D4A', bg: '#FCF3F0', border: '#E8C8BE', Icon: XCircle },
  critical: { label: 'Critical', color: '#9E2A2B', bg: '#FCF0EF', border: '#E8B5B6', Icon: XCircle },
};

function inferMaskKey(name: string, fallback?: string): string {
  if (fallback) return fallback.toLowerCase();
  const n = (name || '').toLowerCase();
  if (n.includes('microaneurysm')) return 'ma';
  if (n.includes('hemorrhage')) return 'he';
  if (n.includes('hard exudate')) return 'ex';
  if (n.includes('soft exudate')) return 'se';
  if (n.includes('dme') || n.includes('macula')) return 'od';
  return '';
}

function toDataUri(base64?: string): string | null {
  if (!base64) return null;
  if (base64.startsWith('data:image')) return base64;
  return `data:image/png;base64,${base64}`;
}

export function ClinicalFeaturesTab({ features, segmentationImages }: ClinicalFeaturesTaProps) {
  const [expandedImage, setExpandedImage] = useState<{ src: string; name: string } | null>(null);

  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-1 gap-4">
        {features.map((f) => {
          const cfg = STATUS_CONFIG[f.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.normal;
          const Icon = cfg.Icon;
          const maskKey = inferMaskKey(f.name, (f as any).mask_key);
          const featureImg = toDataUri(segmentationImages?.feature_images_base64?.[maskKey]);
          const maskImg = toDataUri(segmentationImages?.masks_base64?.[maskKey.toUpperCase()]);
          const imageSrc = featureImg || maskImg || toDataUri(segmentationImages?.combined_overlay_base64);

          return (
            <div
              key={f.name}
              className="rounded-lg border bg-white p-5 transition-all hover:shadow-sm"
              style={{ borderColor: cfg.border }}
            >
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_220px]">
                <div className="flex flex-col gap-3">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="font-semibold text-sm text-[#2C2825]">{f.name}</h3>
                      <p className="text-lg font-bold mt-0.5" style={{ color: cfg.color }}>
                        {f.value}
                        {f.unit && <span className="text-sm font-normal text-[#736D68] ml-1">{f.unit}</span>}
                      </p>
                    </div>
                    <span
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border shrink-0"
                      style={{ color: cfg.color, background: cfg.bg, borderColor: cfg.border }}
                    >
                      <Icon className="w-3 h-3" />
                      {cfg.label}
                    </span>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs text-[#736D68] mb-1">
                      <span>AI Score</span>
                      <span className="font-semibold" style={{ color: cfg.color }}>
                        {(f.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-[#E5DDD6] overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{ width: `${f.score * 100}%`, background: cfg.color }}
                      />
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs text-[#736D68] pt-1 border-t border-[#E5DDD6]">
                    <span>Reference: <strong className="text-[#2C2825]">{f.reference}</strong></span>
                  </div>
                </div>

                <div className="rounded-xl border border-[#E5DDD6] bg-[#FAF8F5] p-2 group relative">
                  <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[#736D68]">Segmentation View</p>
                  {imageSrc ? (
                    <div className="relative">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={imageSrc} alt={`${f.name} segmentation`} className="h-[140px] w-full rounded-lg border border-[#E5DDD6] object-contain bg-white cursor-pointer transition-transform hover:scale-105" onClick={() => setExpandedImage({ src: imageSrc, name: f.name })} />
                      <button
                        onClick={() => setExpandedImage({ src: imageSrc, name: f.name })}
                        className="absolute top-2 right-2 bg-white rounded-lg p-1.5 opacity-0 group-hover:opacity-100 transition-opacity shadow-md hover:bg-[#F7F5F0]"
                        title="Expand view"
                      >
                        <svg className="w-4 h-4 text-[#A85D4A]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6v12h12v-4m7-5V9m0 0l-2.293 2.293m2.293-2.293l2.293 2.293M9 13h6" />
                        </svg>
                      </button>
                    </div>
                  ) : (
                    <div className="flex h-[140px] items-center justify-center rounded-lg border border-dashed border-[#D9D0C7] bg-white text-xs text-[#8B7E74]">
                      Segmentation preview unavailable
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Expanded Image Modal with Dimmed Background */}
      {expandedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          onClick={() => setExpandedImage(null)}
        >
          <div 
            className="bg-white rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-[#E5DDD6] p-6">
              <h2 className="text-xl font-bold text-[#2C2825]">{expandedImage.name} - Enhanced View</h2>
              <button
                onClick={() => setExpandedImage(null)}
                className="h-8 w-8 rounded-lg hover:bg-[#F7F5F0] text-[#736D68] transition-colors flex items-center justify-center"
                title="Close"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 p-6 flex items-center justify-center bg-gradient-to-b from-[#F7F5F0] to-white overflow-auto">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img 
                src={expandedImage.src} 
                alt={expandedImage.name} 
                className="max-w-full max-h-full object-contain rounded-lg border-2 border-[#E5DDD6] shadow-lg"
              />
            </div>
            <div className="border-t border-[#E5DDD6] p-4 bg-[#FCF8F3]">
              <p className="text-sm text-[#736D68]">
                💡 This enhanced view shows the segmentation visualization with dimmed background for better clarity of detected features.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
