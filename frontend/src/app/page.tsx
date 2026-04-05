"use client";

import { useState, useCallback } from "react";
import Chat from "../components/Chat";
import Visualizer from "../components/Visualizer";

export interface ChartConfig {
  id: string;
  title: string;
  type: "bar" | "pie";
  data: any[];
  xKey?: string;
  bars?: { key: string; color: string | null; label: string }[];
}

export interface StepResult {
  analysis: string;
  data_preview: string | null;
  metrics: Record<string, any>;
  charts: ChartConfig[];
}

export default function Home() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<string>("ingestion");
  const [activeTab, setActiveTab] = useState<string>("ingestion");
  const [stepResults, setStepResults] = useState<Record<string, StepResult>>({});
  const [status, setStatus] = useState<string>("idle");

  const handleStateUpdate = useCallback((data: any) => {
    if (data?.session_id && !sessionId) {
      setSessionId(data.session_id);
    }

    if (data?.state) {
      setCurrentStep(data.state);
      setActiveTab(data.state);

      const steps = ["ingestion", "preprocessing", "eda", "standardization", "modeling", "evaluation"];
      const currentIdx = steps.indexOf(data.state);
      const producerStep = currentIdx > 0 ? steps[currentIdx - 1] : data.state;

      if (data.reply) {
        setStepResults(prev => ({
          ...prev,
          [producerStep]: {
            analysis: data.reply,
            data_preview: data.data_preview || null,
            metrics: data.metrics || {},
            charts: data.charts || []
          }
        }));
      }
    }

    if (data?.status) {
      setStatus(data.status);
    }

    if (data?.reply && data?.state === "preprocessing") {
      setStepResults(prev => ({
        ...prev,
        ingestion: {
          analysis: data.reply,
          data_preview: data.data_preview || null,
          metrics: data.metrics || {},
          charts: data.charts || []
        }
      }));
    }

    if (data?.state === "evaluation" && data?.reply) {
      setStepResults(prev => ({
        ...prev,
        evaluation: {
          analysis: data.reply,
          data_preview: data.data_preview || null,
          metrics: data.metrics || {},
          charts: data.charts || []
        }
      }));
    }
  }, [sessionId]);

  return (
    <div className="app-shell">
      <section className="chat-panel">
        <Chat onStateUpdate={handleStateUpdate} sessionId={sessionId} setSessionId={setSessionId} />
      </section>

      <section className="workspace">
        <Visualizer
          currentStep={currentStep}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          stepResults={stepResults}
          status={status}
        />
      </section>
    </div>
  );
}
