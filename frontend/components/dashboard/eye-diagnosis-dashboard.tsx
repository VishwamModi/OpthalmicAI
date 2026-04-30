"use client";

import { useState } from "react";
import {
  Upload, LayoutDashboard, Microscope, BarChart3, TrendingUp,
  FileDown, User, Calendar, Eye, ChevronRight,
} from "lucide-react";

import { MOCK_PATIENT, MOCK_RESULTS } from "./mock-data";
import { UploadScanTab } from "./upload-scan-tab";
import { OverviewTab } from "./overview-tab";
import { ClinicalFeaturesTab } from "./clinical-features-tab";
import { SeverityStagingTab } from "./severity-staging-tab";
import { DecisionCurveTab } from "./decision-curve-tab";
import { PrintReport } from "./print-report";
import { MarketingSection } from "./marketing-section";

type Tab = "upload" | "overview" | "features" | "severity" | "dca";

const TABS: { id: Tab; label: string; icon: React.ElementType; requiresResults?: boolean }[] = [
  { id: "upload", label: "Upload Scan", icon: Upload },
  { id: "overview", label: "Overview", icon: LayoutDashboard, requiresResults: true },
  { id: "features", label: "Clinical Features", icon: Microscope, requiresResults: true },
  { id: "severity", label: "Severity Staging", icon: BarChart3, requiresResults: true },
  { id: "dca", label: "Decision Curve", icon: TrendingUp, requiresResults: true },
];

export function EyeDiagnosisDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [hasResults, setHasResults] = useState(false);

  const handleProcessComplete = () => {
    setHasResults(true);
    setActiveTab("overview");
  };

  const handleExportPDF = () => window.print();

  return (
    <>
      <PrintReport />

      <div className="min-h-screen bg-background no-print">
        {/* ── Top Nav ── */}
        <header className="sticky top-0 z-50 bg-card border-b border-border shadow-sm">
          <div className="max-w-7xl mx-auto px-4 md:px-6 h-14 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <Eye className="w-4 h-4 text-white" />
              </div>
              <div>
                <span className="font-bold text-foreground text-sm">Opthalmic</span>
                <span className="hidden md:inline text-xs text-muted-foreground ml-2">Clinical Diagnosis Platform</span>
              </div>
              <span className="hidden md:inline px-2 py-0.5 rounded-full bg-green-100 text-green-700 text-xs font-semibold border border-green-200">
                Research Prototype
              </span>
            </div>
            <nav className="hidden md:flex items-center gap-1 text-xs text-muted-foreground">
              {["Dashboard", "Patients", "Reports", "Settings"].map((item) => (
                <button key={item} className="px-3 py-1.5 rounded-md hover:bg-muted hover:text-foreground transition-colors">
                  {item}
                </button>
              ))}
            </nav>
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                <User className="w-3.5 h-3.5 text-primary" />
              </div>
            </div>
          </div>
        </header>

        {/* ── Patient Info Bar ── */}
        <div className="bg-card border-b border-border">
          <div className="max-w-7xl mx-auto px-4 md:px-6 py-3 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-bold text-xs">
                {MOCK_PATIENT.name.split(" ").map((n) => n[0]).join("")}
              </div>
              <div>
                <span className="font-semibold text-foreground">{MOCK_PATIENT.name}</span>
                <span className="text-muted-foreground ml-2 text-xs">Age {MOCK_PATIENT.age}</span>
              </div>
            </div>
            <ChevronRight className="w-3.5 h-3.5 text-border hidden sm:block" />
            {[
              { icon: User, label: "ID", value: MOCK_PATIENT.id },
              { icon: Calendar, label: "DOB", value: MOCK_PATIENT.dob },
              { icon: Eye, label: "Eye", value: MOCK_PATIENT.eye },
            ].map((item) => (
              <div key={item.label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <item.icon className="w-3.5 h-3.5" />
                <span className="text-foreground font-medium">{item.value}</span>
              </div>
            ))}

            {hasResults && (
              <div className="ml-auto flex items-center gap-2">
                <span className="flex items-center gap-1.5 text-xs text-green-700 bg-green-50 border border-green-200 px-2.5 py-1 rounded-full font-medium">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  Analysis Complete
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ── Main Diagnostic App ── */}
        <main className="max-w-7xl mx-auto px-4 md:px-6 py-6">
          {/* Tabs + Export */}
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
            {/* Tab bar */}
            <div className="flex items-center gap-1 bg-muted p-1 rounded-xl overflow-x-auto">
              {TABS.map((tab) => {
                const locked = tab.requiresResults && !hasResults;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => !locked && setActiveTab(tab.id)}
                    disabled={locked}
                    className={`
                      flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all
                      ${isActive
                        ? "bg-card text-primary shadow-sm border border-border"
                        : locked
                          ? "text-muted-foreground/40 cursor-not-allowed"
                          : "text-muted-foreground hover:text-foreground hover:bg-card/60"
                      }
                    `}
                  >
                    <tab.icon className={`w-3.5 h-3.5 ${isActive ? "text-primary" : ""}`} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* Export PDF */}
            {hasResults && (
              <button
                onClick={handleExportPDF}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-foreground text-primary-foreground text-xs font-semibold hover:bg-foreground/90 transition-all shadow-sm"
              >
                <FileDown className="w-3.5 h-3.5" />
                Export Clinical PDF
              </button>
            )}
          </div>

          {/* Tab Content */}
          <div>
            {activeTab === "upload" && <UploadScanTab onProcessComplete={handleProcessComplete} />}
            {activeTab === "overview" && hasResults && <OverviewTab />}
            {activeTab === "features" && hasResults && <ClinicalFeaturesTab />}
            {activeTab === "severity" && hasResults && <SeverityStagingTab />}
            {activeTab === "dca" && hasResults && <DecisionCurveTab />}
          </div>

          {/* Results badge shown on overview result */}
          {hasResults && activeTab === "overview" && (
            <div className="mt-6 p-4 rounded-xl border border-primary/20 bg-primary/5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
                  <BarChart3 className="w-4.5 h-4.5 text-primary" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    {MOCK_RESULTS.diagnosis} · {MOCK_RESULTS.etdrsLevel}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    AI Confidence: {MOCK_RESULTS.aiConfidence}% · Model: {MOCK_RESULTS.modelVersion}
                  </p>
                </div>
              </div>
              <span className="px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold">
                Report Ready
              </span>
            </div>
          )}
        </main>

        {/* ── Marketing Section ── */}
        <MarketingSection />
      </div>
    </>
  );
}
