'use client';

import { createPortal } from 'react-dom';

type ReportPatient = {
  id?: string;
  name?: string;
  dob?: string;
  age?: number | string;
  eye?: string;
  examDate?: string;
  referringPhysician?: string;
};

type ReportFeature = {
  name: string;
  value: string;
  unit?: string;
  reference?: string;
  status?: string;
};

type ReportResults = {
  diagnosis?: string;
  icd10?: string;
  etdrsLevel?: string;
  aiConfidence?: number;
  processingTime?: string;
  modelVersion?: string;
  recommendation?: string;
  clinicalFeatures?: ReportFeature[];
  clinical_report?: string;
  patient_report?: string;
  generatedReport?: string;
  segmentation_images?: {
    original_base64?: string;
    combined_overlay_base64?: string;
    masks_base64?: Record<string, string>;
    feature_images_base64?: Record<string, string>;
  };
};

interface PrintReportProps {
  patient?: ReportPatient;
  results?: ReportResults;
  originalImageSrc?: string | null;
  gradcamImageSrc?: string | null;
}

export function PrintReport({
  patient,
  results,
  originalImageSrc,
  gradcamImageSrc,
}: PrintReportProps) {
  const p = patient ?? {};
  const r = results ?? {};
  const rawClinicalText = String((r as any)?.clinical_report || (r as any)?.generatedReport || '').trim();
  const rawPatientText = String((r as any)?.patient_report || '').trim();
  const clinicalText =
    rawClinicalText ||
    `Diagnosis: ${r.diagnosis || 'N/A'} (${r.etdrsLevel || 'N/A'}). Recommendation: ${
      r.recommendation || 'Please follow clinician guidance.'
    }`;
  const patientText =
    rawPatientText ||
    `Your scan suggests ${r.diagnosis || 'an eye finding'}. Please follow the recommended next steps and consult your doctor.`;
  const features = Array.isArray(r?.clinicalFeatures) ? r.clinicalFeatures : [];
  const seg = (r as any)?.segmentation_images;

  const dataUrlFromBase64 = (value?: string) => (value ? `data:image/png;base64,${value}` : null);

  const detailedImages: Array<{ label: string; src: string | null }> = [
    { label: 'Original Retinal Scan', src: originalImageSrc || dataUrlFromBase64(seg?.original_base64) },
    { label: 'Grad-CAM Heatmap', src: gradcamImageSrc },
    { label: 'Segmentation Overlay', src: dataUrlFromBase64(seg?.combined_overlay_base64) },
    { label: 'Optic Disc Mask', src: dataUrlFromBase64(seg?.masks_base64?.OD) },
    { label: 'Microaneurysm Mask', src: dataUrlFromBase64(seg?.masks_base64?.MA) },
    { label: 'Hemorrhage Mask', src: dataUrlFromBase64(seg?.masks_base64?.HE) },
    { label: 'Hard Exudate Mask', src: dataUrlFromBase64(seg?.masks_base64?.EX) },
    { label: 'Soft Exudate Mask', src: dataUrlFromBase64(seg?.masks_base64?.SE) },
  ];

  const report = (
    <div id="print-report" style={{ display: 'none' }}>
      <style>{`
        #print-report { font-family: 'Times New Roman', Times, serif; color: #000; box-sizing: border-box; }
        #print-report * { box-sizing: border-box; }
        @page { size: auto; margin: 12mm; }
        .pr-title { font-size: 18pt; font-weight: 700; text-align: center; margin-bottom: 4px; }
        .pr-subtitle { font-size: 10.5pt; text-align: center; margin-bottom: 16px; color: #111; }
        .pr-table { width: 100%; border-collapse: collapse; font-size: 10.5pt; }
        .pr-table th, .pr-table td { border: 1px solid #000; padding: 6px 10px; text-align: left; vertical-align: top; }
        .pr-table th { background: #f2f2f2; font-weight: 700; }
        .pr-heading { font-size: 12pt; font-weight: 700; margin-bottom: 8px; }
        .pr-box { border: 1px solid #000; padding: 10px 12px; font-size: 10.5pt; line-height: 1.6; }
        .pr-section { margin-bottom: 20px; }
        .pr-avoid-break { break-inside: avoid; page-break-inside: avoid; }
        .pr-table thead { display: table-header-group; }
        .pr-table tfoot { display: table-footer-group; }
        tr, td, th { break-inside: avoid; page-break-inside: avoid; }
        img { display: block; max-width: 100%; height: auto; }
        @media print {
          #print-report { display: block !important; width: 100%; background: white; color-adjust: exact; -webkit-print-color-adjust: exact; }
          .no-print { display: none !important; }
          .print-page-break { break-after: page; page-break-after: always; }
          * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
        }
      `}</style>

      {/* HEADER */}
      <div className="pr-section pr-avoid-break" style={{ borderBottom: '1.5px solid #000', paddingBottom: 10, marginBottom: 16 }}>
        <div className="pr-title">Clinical Retinal Analysis Report</div>
        <div className="pr-subtitle">Automatically generated diagnostic summary</div>
        <table className="pr-table">
          <tbody>
            <tr>
              <td style={{ width: '16%' }}><strong>Patient ID</strong></td>
              <td style={{ width: '34%' }}>{p.id || '—'}</td>
              <td style={{ width: '16%' }}><strong>Name</strong></td>
              <td style={{ width: '34%' }}>{p.name || '—'}</td>
            </tr>
            <tr>
              <td><strong>Date of Birth</strong></td>
              <td>{p.dob || '—'}</td>
              <td><strong>Age</strong></td>
              <td>{p.age ?? '—'}</td>
            </tr>
            <tr>
              <td><strong>Exam Date</strong></td>
              <td>{p.examDate || '—'}</td>
              <td><strong>Eye</strong></td>
              <td>{p.eye || '—'}</td>
            </tr>
            <tr>
              <td><strong>Referring Clinician</strong></td>
              <td colSpan={3}>{p.referringPhysician || '—'}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* SECTION 1 – Primary Diagnosis */}
      <div className="pr-section" style={{ marginBottom: 20 }}>
        <div className="pr-heading">Section 1 — Primary Diagnosis</div>
        <table className="pr-table">
          <tbody>
            <tr>
              <td style={{ width: "30%" }}><strong>Diagnosis</strong></td>
              <td style={{ fontSize: 14, fontWeight: 700 }}>{r.diagnosis || '—'}</td>
            </tr>
            <tr>
              <td><strong>ETDRS Severity</strong></td>
              <td>{r.etdrsLevel || '—'}</td>
            </tr>
            <tr>
              <td><strong>ICD-10 Code</strong></td>
              <td>{r.icd10 || '—'}</td>
            </tr>
            <tr>
              <td><strong>AI Confidence Score</strong></td>
              <td style={{ fontWeight: 700, fontSize: 12 }}>
                {typeof r.aiConfidence === 'number' ? `${r.aiConfidence}%` : '—'}
              </td>
            </tr>
            <tr>
              <td><strong>Model</strong></td>
              <td>
                {r.modelVersion || '—'}
                {r.processingTime ? ` · Processed in ${r.processingTime}` : ''}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* SECTION 2 – Detailed retinal imagery */}
      <div className="pr-section" style={{ marginBottom: 20 }}>
        <div className="pr-heading">Section 2 — Retinal Images</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 12 }}>
          {detailedImages.map(({ label, src }) => (
            <div
              key={label}
              style={{
                minHeight: 180,
                border: '1px solid #000',
                borderRadius: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#fff',
                color: '#222',
                fontSize: 10.5,
                flexDirection: 'column',
                gap: 6,
                overflow: 'hidden',
              }}
            >
              {src ? (
                // eslint-disable-next-line @next/next/no-img-element
                <>
                  <img src={src} alt={label} style={{ width: '100%', height: 145, objectFit: 'contain' }} />
                  <div style={{ width: '100%', borderTop: '1px solid #000', padding: '4px 8px', fontWeight: 600 }}>{label}</div>
                </>
              ) : (
                <>
                  <div style={{ width: 32, height: 32, border: '1.5px solid #94a3b8', borderRadius: 4 }} />
                  {label}
                </>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* SECTION 3 – Clinical Features */}
      <div className="pr-section" style={{ marginBottom: 20 }}>
        <div className="pr-heading">Section 3 — Clinical Features</div>
        <table className="pr-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Detected Value</th>
              <th>Reference Range</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {features.map((f) => (
              <tr key={f.name}>
                <td>{f.name}</td>
                <td>{f.value} {f.unit}</td>
                <td>{f.reference}</td>
                <td style={{ textTransform: "capitalize", fontWeight: 600 }}>{f.status}</td>
              </tr>
            ))}
            {!features.length && (
              <tr>
                <td colSpan={4}>No extracted clinical features available.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* SECTION 4 – Recommendation & Signature */}
      <div className="pr-section" style={{ marginBottom: 24 }}>
        <div className="pr-heading">Section 4 — Clinical Recommendation</div>
        <div className="pr-box">
          {r.recommendation || 'No recommendation available.'}
        </div>
      </div>

      {/* SECTION 5 – Dual LLM Reports */}
      <div className="pr-section" style={{ marginBottom: 24 }}>
        <div className="pr-heading">Section 5 — AI Narrative Reports</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
          <div style={{ border: "1px solid #000", borderRadius: 0, padding: "12px 16px", background: "#fff" }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Clinical View</div>
            <div style={{ whiteSpace: "pre-wrap", fontSize: 10.5, lineHeight: 1.6, color: "#000" }}>
              {clinicalText || "Generating clinical report..."}
            </div>
          </div>
          <div style={{ border: "1px solid #000", borderRadius: 0, padding: "12px 16px", background: "#fff" }}>
            <div style={{ fontWeight: 700, marginBottom: 6 }}>Patient View</div>
            <div style={{ whiteSpace: "pre-wrap", fontSize: 10.5, lineHeight: 1.6, color: "#000" }}>
              {patientText || "Generating patient-friendly report..."}
            </div>
          </div>
        </div>
      </div>

      {/* Signature */}
      <div className="pr-section pr-avoid-break" style={{ marginTop: 36, borderTop: "1px solid #000", paddingTop: 16, display: "flex", justifyContent: "space-between", fontSize: 10.5, color: "#000" }}>
        <div>
          <div>Physician Signature: ___________________________________</div>
          <div style={{ marginTop: 6 }}>Date: _______________</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div>This report was generated by the analysis platform.</div>
          <div>For clinical review only.</div>
        </div>
      </div>
    </div>
  );

  if (typeof document === 'undefined') {
    return report;
  }

  return createPortal(report, document.body);
}
