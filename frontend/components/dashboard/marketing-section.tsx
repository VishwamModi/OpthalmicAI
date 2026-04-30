"use client";

import { ShieldCheck, Database, Award } from "lucide-react";
import { Camera, Cloud, FileText } from "lucide-react";

const VALIDATION_METRICS = [
  {
    icon: ShieldCheck,
    value: "12.2M",
    label: "EfficientNet-B3 Params",
    sub: "Compound-scaled backbone for retinal lesion detail",
    color: "#0284C7",
    bg: "#eff6ff",
    border: "#bfdbfe",
  },
  {
    icon: Database,
    value: "35,108 + 3,662",
    label: "EyePACS + APTOS",
    sub: "Two-stage transfer learning pipeline",
    color: "#0D9488",
    bg: "#f0fdfa",
    border: "#99f6e4",
  },
  {
    icon: Award,
    value: "Research",
    label: "No FDA Clearance Yet",
    sub: "Current build is for research/demo workflows",
    color: "#7c3aed",
    bg: "#f5f3ff",
    border: "#ddd6fe",
  },
];

const HOW_IT_WORKS = [
  {
    step: 1,
    icon: Camera,
    title: "Capture",
    desc: "Upload a retinal fundus image and apply circular masking plus CLAHE-enhanced green-channel preprocessing.",
    color: "#0284C7",
    bg: "#eff6ff",
  },
  {
    step: 2,
    icon: Cloud,
    title: "Cloud AI Analysis",
    desc: "EfficientNet-B3 predicts a continuous DR severity score using Smooth L1 + MSE loss and progressive unfreezing.",
    color: "#0D9488",
    bg: "#f0fdfa",
  },
  {
    step: 3,
    icon: FileText,
    title: "Decision-Support Report",
    desc: "Generate DR grade output and confidence context for clinician review; runtime varies by hardware.",
    color: "#7c3aed",
    bg: "#f5f3ff",
  },
];

export function MarketingSection() {
  return (
    <section className="border-t border-border bg-background">
      {/* Clinical Validation */}
      <div className="max-w-6xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <span className="inline-block px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold mb-4 border border-primary/20">
            Clinical Evidence
          </span>
          <h2 className="text-3xl font-bold text-foreground text-balance">
            Built on Rigorous Clinical Validation
          </h2>
          <p className="text-muted-foreground mt-3 max-w-xl mx-auto leading-relaxed text-balance">
            Opthalmic currently reflects a research-stage DR pipeline with documented architecture, preprocessing, and transfer-learning strategy.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {VALIDATION_METRICS.map((m) => (
            <div
              key={m.label}
              className="rounded-2xl border p-8 text-center flex flex-col items-center gap-3 hover:shadow-md transition-shadow"
              style={{ background: m.bg, borderColor: m.border }}
            >
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center"
                style={{ background: "white", border: `1.5px solid ${m.border}` }}
              >
                <m.icon className="w-7 h-7" style={{ color: m.color }} />
              </div>
              <div className="text-4xl font-black" style={{ color: m.color }}>{m.value}</div>
              <div className="font-semibold text-foreground">{m.label}</div>
              <div className="text-xs text-muted-foreground">{m.sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className="bg-muted/40 border-t border-border">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="text-center mb-12">
            <span className="inline-block px-3 py-1 rounded-full bg-accent/10 text-accent text-xs font-semibold mb-4 border border-accent/20">
              Workflow
            </span>
            <h2 className="text-3xl font-bold text-foreground text-balance">How It Works</h2>
            <p className="text-muted-foreground mt-3 max-w-xl mx-auto leading-relaxed text-balance">
              From fundus image to DR severity estimate with explainable model outputs.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connector line (desktop) */}
            <div className="hidden md:block absolute top-14 left-[calc(33%-16px)] right-[calc(33%-16px)] h-0.5 bg-border" />

            {HOW_IT_WORKS.map((s) => (
              <div key={s.step} className="flex flex-col items-center text-center gap-4">
                <div className="relative">
                  <div
                    className="w-28 h-28 rounded-2xl flex items-center justify-center shadow-sm border"
                    style={{ background: s.bg, borderColor: s.border ?? "#e2e8f0" }}
                  >
                    <s.icon className="w-12 h-12" style={{ color: s.color }} />
                  </div>
                  <div
                    className="absolute -top-2 -right-2 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white border-2 border-background"
                    style={{ background: s.color }}
                  >
                    {s.step}
                  </div>
                </div>
                <div>
                  <h3 className="font-bold text-foreground text-lg">{s.title}</h3>
                  <p className="text-sm text-muted-foreground mt-1 leading-relaxed">{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer bar */}
      <div className="border-t border-border bg-card">
        <div className="max-w-6xl mx-auto px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2 font-semibold text-foreground">
            <div className="w-6 h-6 rounded-md bg-primary flex items-center justify-center">
              <span className="text-white text-xs font-black">O</span>
            </div>
            Opthalmic Clinical Platform
          </div>
          <div className="flex flex-wrap gap-6 justify-center">
            {["Research Prototype", "Model Metrics Pending External Validation", "Regulatory Pathway In Progress", "Security & Compliance Roadmap"].map((badge) => (
              <span key={badge} className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-primary" />
                {badge}
              </span>
            ))}
          </div>
          <div>© 2026 Opthalmic. All rights reserved.</div>
        </div>
      </div>
    </section>
  );
}
