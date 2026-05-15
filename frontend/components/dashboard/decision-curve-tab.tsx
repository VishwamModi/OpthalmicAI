'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { MOCK_RESULTS } from './mock-data';

interface DecisionCurveTabProps {
  data: typeof MOCK_RESULTS['dcaCurveData'];
}

export function DecisionCurveTab({ data }: DecisionCurveTabProps) {
  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-lg border border-[#E5DDD6] bg-white p-6">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-6">
          <div>
            <h3 className="font-semibold text-[#2C2825]">Decision Curve Analysis</h3>
            <p className="text-sm text-[#736D68] mt-1">
              Net benefit comparison across threshold probabilities
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Opthalmic Model', color: '#A85D4A' },
              { label: 'Treat All', color: '#A39A8E' },
              { label: 'Treat None', color: '#D4CCC2' },
            ].map((l) => (
              <div key={l.label} className="flex items-center gap-1.5 text-xs text-[#736D68]">
                <div className="w-4 h-0.5 rounded" style={{ background: l.color }} />
                {l.label}
              </div>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5DDD6" />
            <XAxis
              dataKey="threshold"
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 11, fill: '#736D68' }}
              label={{ value: 'Threshold Probability', position: 'insideBottomRight', offset: -8, fontSize: 11, fill: '#736D68' }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#736D68' }}
              label={{ value: 'Net Benefit', angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#736D68' }}
            />
            <Tooltip
              contentStyle={{ background: 'white', border: '1px solid #E5DDD6', borderRadius: 8, fontSize: 12 }}
              formatter={(value: number, name: string) => [value.toFixed(3), name]}
              labelFormatter={(v) => `Threshold: ${(Number(v) * 100).toFixed(0)}%`}
            />
            <ReferenceLine y={0} stroke="#E5DDD6" />
            <Line
              type="monotone" dataKey="model" name="Opthalmic Model"
              stroke="#A85D4A" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }}
            />
            <Line
              type="monotone" dataKey="allTreat" name="Treat All"
              stroke="#A39A8E" strokeWidth={1.5} dot={false} strokeDasharray="4 4"
            />
            <Line
              type="monotone" dataKey="netBenefit" name="Treat None"
              stroke="#D4CCC2" strokeWidth={1.5} dot={false} strokeDasharray="2 3"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Interpretation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          {
            title: 'Model Superiority Zone',
            desc: 'Opthalmic demonstrates superior net benefit compared to both treat-all and treat-none strategies across the clinically relevant threshold range (10–60%).',
            color: '#A85D4A',
            bg: '#FCF3F0',
            border: '#E8C8BE',
          },
          {
            title: 'AUC-DCA Interpretation',
            desc: 'The broad positive net benefit plateau indicates the model is well-calibrated for clinical decision support, minimizing both over-treatment and missed diagnoses.',
            color: '#A39A8E',
            bg: '#F2EFE9',
            border: '#D8CFCA',
          },
        ].map((item) => (
          <div
            key={item.title}
            className="rounded-lg border p-4"
            style={{ background: item.bg, borderColor: item.border }}
          >
            <h4 className="font-semibold text-sm mb-1" style={{ color: item.color }}>{item.title}</h4>
            <p className="text-xs text-[#736D68] leading-relaxed text-justify">{item.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
