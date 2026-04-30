'use client';

import { Eye } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

export function HomePage({ onNavigate }: { onNavigate: (view: string) => void }) {
  const galleryRef = useRef<HTMLDivElement | null>(null);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);

  const galleryImages = [
    { filename: 'normal.jpeg', description: 'No DR' },
    { filename: 'mild.jpeg', description: 'Mild NPDR' },
    { filename: 'moderate.jpeg', description: 'Moderate NPDR' },
    { filename: 'severe.jpeg', description: 'Severe NPDR' },
    { filename: 'proliferative.jpeg', description: 'Proliferative DR' },
    { filename: 'grad_cam.png', description: 'Grad-CAM Heatmap' },
    { filename: 'DCA_analysis.png', description: 'Decision Curve Analysis' },
    { filename: 'report_pg1.png', description: 'Clinical Report Page 1' },
    { filename: 'report_pg2.png', description: 'Clinical Report Page 2' },
    { filename: 'report_pg3.png', description: 'Clinical Report Page 3' },
    { filename: 'fundus_in_use.jpg', description: 'Eye Examination Setup' },
    { filename: 'fundus-camera.jpg', description: 'Fundus Camera Device' },
  ];

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  useEffect(() => {
    if (!autoScrollEnabled) return;
    let frame = 0;

    const tick = () => {
      const scroller = galleryRef.current;
      if (!scroller) {
        frame = requestAnimationFrame(tick);
        return;
      }

      const resetAt = scroller.scrollWidth / 2;
      scroller.scrollLeft += 0.45;

      if (scroller.scrollLeft >= resetAt) {
        scroller.scrollLeft = 0;
      }

      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [autoScrollEnabled]);

  return (
    <div className="w-full bg-[#FCF8F3] text-[#3B2416]">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-[#FCF8F3]/95 border-b border-[#E8D9C8] shadow-sm backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#3B2416]">
              <Eye className="h-5 w-5 text-white" />
            </div>
            <div>
              <p className="text-lg font-bold text-[#3B2416] leading-none">Opthalmic</p>
              <p className="text-xs text-[#6E4B34] mt-1">Clinical Diagnosis</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <a href="#about" onClick={(e) => { e.preventDefault(); scrollToSection('about'); }} className="text-[#3B2416] hover:text-[#6E4B34] text-sm font-medium transition-colors">About Us</a>
            <a href="#how-it-works" onClick={(e) => { e.preventDefault(); scrollToSection('how-it-works'); }} className="text-[#3B2416] hover:text-[#6E4B34] text-sm font-medium transition-colors">How It Works</a>
            <a href="#video" onClick={(e) => { e.preventDefault(); scrollToSection('video'); }} className="text-[#3B2416] hover:text-[#6E4B34] text-sm font-medium transition-colors">Video</a>
            <a href="#contact" onClick={(e) => { e.preventDefault(); scrollToSection('contact'); }} className="text-[#3B2416] hover:text-[#6E4B34] text-sm font-medium transition-colors">Contact Us</a>
            <button onClick={() => onNavigate('login')} className="px-6 py-2 bg-[#3B2416] text-white rounded-lg text-sm font-semibold hover:bg-[#2F1B11]">Open Portal</button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section
        className="relative w-full h-[34rem] md:h-[40rem] bg-cover bg-center flex items-center justify-center"
        style={{ backgroundImage: 'url(/images/hero-eye.jpg)', backgroundSize: '100%', backgroundPosition: 'center 40%' }}
      >
        <div className="absolute inset-0 bg-black/45" />
        <div className="relative z-10 max-w-4xl mx-auto text-center px-6">
          <p className="text-sm font-semibold text-gray-100 uppercase tracking-wider mb-2">Welcome to Opthalmic</p>
          <h2 className="text-5xl font-bold text-white mb-4 max-w-3xl mx-auto">AI-Assisted Diabetic Retinopathy Grading</h2>
          <p className="text-lg text-gray-100 mb-6 max-w-2xl mx-auto">EfficientNet-B3 based fundus analysis for research-grade DR staging and clinician decision support.</p>
          <button onClick={() => onNavigate('login')} className="px-8 py-3 bg-[#3B2416] text-white font-semibold rounded-lg hover:bg-[#2F1B11] inline-block">Get Started</button>
        </div>
      </section>

      {/* About */}
      <section id="about" className="scroll-mt-24 py-20 px-6 bg-[#FCF8F3]">
        <div className="max-w-6xl mx-auto">
          <p className="text-sm font-semibold uppercase tracking-wider text-[#6E4B34] text-center mb-3">About</p>
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-[#3B2416] text-center mb-4">Clinical Intelligence Stack</h2>
          <p className="text-center text-[#6E4B34] leading-relaxed max-w-3xl mx-auto mb-12">Opthalmic combines DR staging, lesion segmentation, explainability, decision-curve support, structured notes, and professional PDF reporting for each scan in one continuous workflow.</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white rounded-lg p-8 border border-[#E8D9C8]">
              <h3 className="text-xl font-bold text-[#3B2416] mb-3">DR Staging</h3>
              <p className="text-[#6E4B34] leading-relaxed">EfficientNet-B3 with fundus preprocessing (masking + CLAHE) estimates ETDRS severity and confidence for triage support.</p>
            </div>
            <div className="bg-white rounded-lg p-8 border border-[#E8D9C8]">
              <h3 className="text-xl font-bold text-[#3B2416] mb-3">Segmentation</h3>
              <p className="text-[#6E4B34] leading-relaxed">Dedicated lesion models generate OD, MA, HE, EX, and SE masks to derive objective feature counts, areas, and DME-linked indicators.</p>
            </div>
            <div className="bg-white rounded-lg p-8 border border-[#E8D9C8]">
              <h3 className="text-xl font-bold text-[#3B2416] mb-3">Decision Support</h3>
              <p className="text-[#6E4B34] leading-relaxed">Grad-CAM, DCA curves, clinical notes, and dual narratives (clinical + patient) are generated automatically per scan.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Workflow */}
      <section id="how-it-works" className="scroll-mt-24 py-20 px-6 bg-white border-t border-[#E8D9C8]">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <p className="text-sm font-semibold uppercase tracking-wider text-[#6E4B34] mb-3">Workflow</p>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-[#3B2416] mb-4">How It Works</h2>
            <p className="text-[#6E4B34] leading-relaxed max-w-3xl mx-auto">A full scan lifecycle from upload and segmentation to risk interpretation, notes, and printable reporting.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-16">
            <div className="flex flex-col items-center text-center">
              <div className="w-48 h-48 rounded-lg bg-[#E7C7B2] mb-6 overflow-hidden border-2 border-[#3B2416]">
                <img src="/images/fundus-camera.jpg" alt="Capture" className="w-full h-full object-cover" />
              </div>
              <h4 className="text-xl font-bold text-[#3B2416] mb-3">Capture</h4>
              <p className="text-sm text-[#6E4B34]">Upload a fundus scan and run retinal validation, masking, and normalization.</p>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="w-48 h-48 rounded-lg bg-[#F4EADD] mb-6 overflow-hidden border-2 border-[#3B2416]">
                <img src="/images/analysis-vision.svg" alt="Analysis" className="w-full h-full object-cover" />
              </div>
              <h4 className="text-xl font-bold text-[#3B2416] mb-3">Stage</h4>
              <p className="text-sm text-[#6E4B34]">DR severity and confidence are estimated with ETDRS-level mapping.</p>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="w-48 h-48 rounded-lg bg-[#F4EADD] mb-6 overflow-hidden border-2 border-[#3B2416]">
                <img src="/images/proliferative.jpeg" alt="Segmentation" className="w-full h-full object-cover" />
              </div>
              <h4 className="text-xl font-bold text-[#3B2416] mb-3">Segment</h4>
              <p className="text-sm text-[#6E4B34]">Lesion masks and overlays quantify MA, HE, EX, SE, and OD features.</p>
            </div>
            <div className="flex flex-col items-center text-center">
              <div className="w-48 h-48 rounded-lg bg-white mb-6 overflow-hidden border-2 border-[#3B2416]">
                <img src="/images/report_pg1.png" alt="Report" className="w-full h-full object-cover" />
              </div>
              <h4 className="text-xl font-bold text-[#3B2416] mb-3">Report</h4>
              <p className="text-sm text-[#6E4B34]">Structured reports, technical notes, DCA interpretation, and PDF export are generated per scan.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Video Section */}
      <section id="video" className="scroll-mt-24 py-20 px-6 bg-[#3B2416]">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <p className="text-sm font-semibold uppercase tracking-wider text-[#D1D5DB] mb-3">Watch</p>
            <h2 className="text-4xl font-bold text-white mb-4">Platform Walkthrough</h2>
            <p className="text-gray-300 max-w-2xl mx-auto">
              Watch the full journey from upload to interpretability and final report generation.
            </p>
          </div>

          <div className="bg-black rounded-lg overflow-hidden shadow-2xl aspect-video">
            <img src="/images/fundus_in_use.jpg" alt="Platform workflow preview" className="w-full h-full object-cover" />
          </div>

          <div className="mt-8 text-center">
            <button onClick={() => onNavigate('login')} className="px-6 py-3 bg-white text-[#3B2416] font-semibold rounded-lg hover:bg-[#F4EADD]">
              Open Patient Data Portal
            </button>
          </div>
        </div>
      </section>

      {/* Gallery */}
      <section className="py-20 px-0 bg-[#FCF8F3]">
        <div className="px-6 mb-12">
          <div className="max-w-6xl mx-auto text-center">
            <p className="text-sm font-semibold uppercase tracking-wider text-[#6E4B34] mb-3">Visual Showcase</p>
            <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-[#3B2416] mb-4">Clinical Gallery</h2>
            <p className="text-[#6E4B34] leading-relaxed max-w-3xl mx-auto">
              Explore disease stages, explainability visualizations, decision curve analysis, and comprehensive clinical reports.
            </p>
          </div>
        </div>

        <div className="relative group">
          <button
            onClick={() => {
              if (galleryRef.current) {
                galleryRef.current.scrollBy({ left: -400, behavior: 'smooth' });
              }
            }}
            className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-[#A85D4A] hover:bg-[#935246] text-white rounded-r-lg p-3 transition-all opacity-0 group-hover:opacity-100"
            aria-label="Scroll left"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>

          <div
            ref={galleryRef}
            onMouseDown={() => setAutoScrollEnabled(false)}
            onWheel={() => setAutoScrollEnabled(false)}
            onTouchStart={() => setAutoScrollEnabled(false)}
            className="flex gap-4 overflow-x-auto pb-4 snap-x snap-mandatory [scrollbar-width:none] [&::-webkit-scrollbar]:hidden px-6"
          >
            {galleryImages.map((item, idx) => (
              <div key={`${item.filename}-${idx}`} className="min-w-[280px] md:min-w-[340px] aspect-[16/10] rounded-lg overflow-hidden border-2 border-[#E8D9C8] snap-start transition-all duration-300 hover:-translate-y-2 hover:shadow-2xl hover:border-[#6E4B34] bg-white flex flex-col">
                <img src={`/images/${item.filename}`} alt={item.description} className="flex-1 w-full object-cover transition-transform duration-300 hover:scale-105" />
                <div className="bg-white px-3 py-2 border-t border-[#E8D9C8]">
                  <p className="text-xs font-semibold text-[#6E4B34] text-center">{item.description}</p>
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={() => {
              if (galleryRef.current) {
                galleryRef.current.scrollBy({ left: 400, behavior: 'smooth' });
              }
            }}
            className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-[#A85D4A] hover:bg-[#935246] text-white rounded-l-lg p-3 transition-all opacity-0 group-hover:opacity-100"
            aria-label="Scroll right"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </section>

      <section id="contact" className="scroll-mt-24 py-20 px-6 bg-[#3B2416] text-white">
        <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-[1fr_1.12fr_1fr] gap-6 auto-rows-fr items-stretch">
          <div className="rounded-2xl bg-white/10 border border-white/20 p-8 md:p-9 backdrop-blur-sm h-full flex flex-col">
            <p className="text-sm font-semibold uppercase tracking-wider text-[#D1D5DB] mb-3">Contact</p>
            <h2 className="text-4xl font-bold tracking-tight mb-4">Get in Touch</h2>
            <div className="space-y-4 text-base text-gray-200">
              <div className="flex items-center justify-between border-b border-white/15 pb-2">
                <span>Research discussions</span>
                <span className="font-semibold">Open</span>
              </div>
              <div className="flex items-center justify-between border-b border-white/15 pb-2">
                <span>Demo requests</span>
                <span className="font-semibold">Available</span>
              </div>
              <div className="flex items-center justify-between border-b border-white/15 pb-2">
                <span>Technical review</span>
                <span className="font-semibold">Supported</span>
              </div>
            </div>
            <p className="mt-auto pt-6 text-base text-gray-200">Response window: within 24–48 hours.</p>
          </div>

          <div className="rounded-2xl bg-[#FCF8F3] text-[#3B2416] border border-[#E8D9C8] p-8 md:p-9 shadow-xl h-full flex flex-col">
            <p className="text-sm font-semibold uppercase tracking-wider text-[#6E4B34] mb-3">Contributors & Faculty</p>
            <div className="space-y-4">
              {[
                { name: 'Vishwam Modi', email: 'vishwam.mce22@sot.pdpu.ac.in' },
                { name: 'Aarchi Shah', email: 'aarchi.sce22@sot.pdpu.ac.in' },
                { name: 'Swati Nandha', email: 'swati.nce22@sot.pdpu.ac.in' },
                { name: 'Dr. Yogesh Kumar', email: 'yogesh.kumar@sot.pdpu.ac.in' },
              ].map((person) => (
                <div key={person.email} className="rounded-xl border border-[#E8D9C8] bg-white p-4">
                  <h3 className="text-xl font-semibold tracking-tight">{person.name}</h3>
                  <a
                    href={`mailto:${person.email}`}
                    className="mt-1 inline-block text-base font-medium text-[#3B2416] hover:text-[#6E4B34] underline underline-offset-4 break-words"
                  >
                    {person.email}
                  </a>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl bg-white/10 border border-white/20 p-8 md:p-9 backdrop-blur-sm h-full flex flex-col">
            <p className="text-sm font-semibold uppercase tracking-wider text-gray-300 mb-3">Collaboration</p>
            <h3 className="text-4xl font-bold tracking-tight mb-4">Project Readiness</h3>
            <div className="space-y-4 text-base text-gray-200 mt-auto">
              <div className="flex items-center justify-between border-b border-white/15 pb-2">
                <span>Model explainability</span>
                <span className="font-semibold">Enabled</span>
              </div>
              <div className="flex items-center justify-between border-b border-white/15 pb-2">
                <span>Report export pipeline</span>
                <span className="font-semibold">Available</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Clinical discussion support</span>
                <span className="font-semibold">Open</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <footer className="bg-[#3B2416] text-white py-10 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div>
            <h3 className="font-bold text-lg">Opthalmic</h3>
            <p className="text-sm text-gray-400 mt-1">Clinical Diagnosis Platform</p>
          </div>
          <p className="text-sm text-gray-400">© 2026 Opthalmic. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
