'use client';

import React, { useState } from 'react';
import { apiUrl } from '@/lib/api-base';

interface UploadScanTabProps {
  onProcessComplete: (results: any) => void;
}

export function UploadScanTab({ onProcessComplete }: UploadScanTabProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const processingStages = [
    'Enhancing contrast via CLAHE...',
    'Running EfficientNet-B3...',
    'Generating Grad-CAM...',
    'Compiling clinical report...',
  ];

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) {
      setFile(dropped);
      void startProcessing(dropped);
    }
  };

  const startProcessing = async (selected: File) => {
    setProcessing(true);
    setProcessingStage(0);
    setError(null);

    let interval: any = null;
    interval = setInterval(() => {
      setProcessingStage((prev) => Math.min(prev + 1, processingStages.length - 1));
    }, 750);

    try {
      const fd = new FormData();
      fd.append('file', selected);
      const res = await fetch(apiUrl('/api/analyze'), { method: 'POST', body: fd });
      const json = (await res.json()) as any;
      if (!res.ok || json?.status !== 'success') {
        throw new Error(json?.message || 'Analysis failed');
      }
      const data = json?.data;
      if (!data || !data.diagnosis) {
        throw new Error('Malformed response from server (missing diagnosis field).');
      }

      onProcessComplete(data);
    } catch (err: any) {
      setError(err?.message || 'Upload failed');
    } finally {
      if (interval) clearInterval(interval);
      setTimeout(() => setProcessing(false), 500);
    }
  };

  if (processing) {
    return (
      <div className="flex flex-col items-center justify-center py-24 px-6">
        <div className="relative w-48 h-48 mb-8">
          <svg className="w-full h-full" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="95" fill="#f0ede8" stroke="#d4c5ba" strokeWidth="2" />
            <circle cx="100" cy="100" r="85" fill="#f5f3f0" />
            <circle cx="85" cy="100" r="20" fill="#e8d5c4" opacity="0.6" />
            <circle cx="110" cy="115" r="12" fill="#dcc8b8" opacity="0.5" />
          </svg>
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: `linear-gradient(to bottom, 
                transparent 0%, 
                rgba(168, 93, 74, 0.3) 48%, 
                rgba(168, 93, 74, 0.6) 50%, 
                rgba(168, 93, 74, 0.3) 52%, 
                transparent 100%)`,
              animation: 'scan 2s ease-in-out infinite',
            }}
          />
          <style>{`
            @keyframes scan {
              0% { transform: translateY(-100px); }
              50% { transform: translateY(0); }
              100% { transform: translateY(100px); }
            }
          `}</style>
        </div>

        <div className="text-center">
          <p className="text-lg font-semibold text-[#A85D4A] mb-2">Processing...</p>
          <p className="text-sm text-[#736D68]">{processingStages[processingStage]}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto py-8 flex flex-col gap-6">
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`
          relative flex flex-col items-center justify-center gap-5
          rounded-lg border-2 border-dashed p-14 cursor-pointer
          transition-all duration-200
          ${isDragging
            ? "border-[#A85D4A] bg-[#A85D4A]/5"
            : file
              ? "border-[#6B705C] bg-[#6B705C]/5"
              : "border-[#E5DDD6] bg-[#F7F5F0] hover:bg-[#EBE6E1]"
          }
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,.dcm"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.[0]) {
              const f = e.target.files[0];
              setFile(f);
            }
          }}
        />

        {file ? (
          <>
            <div className="text-4xl">✓</div>
            <div className="text-center">
              <p className="text-lg font-semibold text-[#2C2825]">{file.name}</p>
              <p className="text-sm text-[#736D68] mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} MB · Ready to process
              </p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setFile(null); }}
              className="text-xs text-[#736D68] underline underline-offset-2 hover:text-[#2C2825] transition-colors"
            >
              Remove file
            </button>
          </>
        ) : (
          <>
            <div className="w-20 h-20 rounded-lg bg-[#A85D4A]/10 flex items-center justify-center">
              <span className="text-3xl">📷</span>
            </div>
            <div className="text-center">
              <p className="text-base font-semibold text-[#2C2825]">
                Drag &amp; drop retinal fundus image here
              </p>
              <p className="text-sm text-[#736D68] mt-1">or click to browse</p>
              <div className="flex items-center justify-center gap-3 mt-4">
                {["PNG", "JPG", "TIFF", "DICOM"].map((fmt) => (
                  <span key={fmt} className="px-2 py-0.5 rounded-full bg-[#EBE6E1] text-xs font-medium text-[#736D68] border border-[#E5DDD6]">
                    {fmt}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}

        {!file && (
          <div className="absolute bottom-3 right-4 flex items-center gap-1.5 text-xs text-[#736D68]">
            📊 Supports standard fundus cameras &amp; OCT
          </div>
        )}
      </div>

      {!processing && file && (
        <button
          onClick={() => void startProcessing(file)}
          className="w-full py-3 rounded-lg font-semibold bg-[#A85D4A] hover:bg-[#935246] text-white transition-colors"
        >
          Process Image
        </button>
      )}

      {error && (
        <div className="rounded-lg border border-[#E8C8BE] bg-[#FCF3F0] px-4 py-3 text-sm text-[#A85D4A]">
          {error}
        </div>
      )}

      {!file && !processing && (
        <button
          onClick={() => {
            const demoFile = new File(["demo"], "retina_fundus_OD.png", { type: "image/png" });
            setFile(demoFile);
            setProcessing(true);
            setProcessingStage(0);

            const interval = setInterval(() => {
              setProcessingStage((prev) => {
                if (prev >= processingStages.length - 1) {
                  clearInterval(interval);
                  setTimeout(() => {
                    setProcessing(false);
                    setError("Demo image is disabled in live mode. Please upload a real file.");
                  }, 500);
                  return prev;
                }
                return prev + 1;
              });
            }, 750);
          }}
          className="text-sm text-[#A85D4A] underline underline-offset-2 hover:text-[#935246] transition-colors text-center"
        >
          Use demo image instead
        </button>
      )}
    </div>
  );
}
