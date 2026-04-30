'use client';

import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type DCADataPoint = {
  threshold: number;
  model: number;
  treatAll: number;
  treatNone: number;
};

interface DecisionCurveAnalysisProps {
  data: DCADataPoint[];
}

export function DecisionCurveAnalysis({ data }: DecisionCurveAnalysisProps) {
  const chartData = useMemo(() => data || [], [data]);

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <h3 className="text-lg font-semibold text-slate-900">Decision Curve Analysis</h3>
            <p className="mt-1 text-sm leading-6 text-slate-500">
              This chart compares the OphthalmicAI model against two clinical baselines. The model is most useful where its
              line stays above both baselines, because that indicates a higher net clinical benefit at the same risk threshold.
            </p>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
            Thresholds: 0.0 → 1.0 in 0.05 steps
          </div>
        </div>

        <div className="mt-6 h-[380px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 12, right: 24, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis
                dataKey="threshold"
                tickFormatter={(value) => `${(Number(value) * 100).toFixed(0)}%`}
                tick={{ fontSize: 11, fill: '#64748B' }}
                label={{
                  value: 'Threshold Probability (Risk Tolerance)',
                  position: 'insideBottom',
                  offset: -4,
                  fontSize: 11,
                  fill: '#64748B',
                }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748B' }}
                label={{
                  value: 'Net Clinical Benefit',
                  angle: -90,
                  position: 'insideLeft',
                  offset: 8,
                  fontSize: 11,
                  fill: '#64748B',
                }}
              />
              <Tooltip
                contentStyle={{
                  background: '#FFFFFF',
                  border: '1px solid #E2E8F0',
                  borderRadius: 14,
                  boxShadow: '0 10px 30px rgba(15, 23, 42, 0.08)',
                }}
                formatter={(value: number, name: string) => [value.toFixed(4), name]}
                labelFormatter={(value) => `Threshold: ${(Number(value) * 100).toFixed(0)}%`}
              />
              <Legend
                verticalAlign="top"
                align="right"
                formatter={(value) => <span style={{ color: '#475569' }}>{value}</span>}
              />
              <ReferenceLine y={0} stroke="#CBD5E1" />
              <Line
                type="monotone"
                dataKey="model"
                name="OphthalmicAI Model"
                stroke="#0F766E"
                strokeWidth={3.5}
                dot={false}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="treatAll"
                name="Treat All"
                stroke="#94A3B8"
                strokeWidth={2.25}
                dot={false}
                strokeDasharray="7 5"
              />
              <Line
                type="monotone"
                dataKey="treatNone"
                name="Treat None"
                stroke="#334155"
                strokeWidth={2.25}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {[
          {
            title: 'Model line',
            body: 'The teal curve represents the net clinical benefit of using the OphthalmicAI model at each threshold.',
            accent: '#0F766E',
          },
          {
            title: 'Treat All',
            body: 'The dashed grey curve shows the baseline where every patient is treated or referred.',
            accent: '#94A3B8',
          },
          {
            title: 'Treat None',
            body: 'The dark grey line is the zero-benefit baseline where no one is treated.',
            accent: '#334155',
          },
        ].map((card) => (
          <div key={card.title} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: card.accent }} />
              {card.title}
            </div>
            <p className="mt-2 text-sm leading-6 text-slate-500">{card.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
