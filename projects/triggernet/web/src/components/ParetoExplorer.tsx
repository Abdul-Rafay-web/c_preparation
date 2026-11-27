import { useMemo, useState } from 'react';
import { Sparkles } from 'lucide-react';
import type { DashboardData } from '../types';

interface Props {
  data: DashboardData;
}

interface PlotPoint {
  tag: string;
  bits: number;
  sparsityPct: number;
  bops: number;
  bopReductionPct: number;
  acc: number;
  auc: number;
  rej50: number;
}

const BIT_COLORS: Record<number, string> = {
  32: '#94a3b8',
  16: '#60a5fa',
  12: '#2dd4bf',
  8: '#38e8ff',
  6: '#f2b84b',
  4: '#f2735b',
};

const ROC_COLORS: Record<string, string> = {
  g: '#94a3b8',
  q: '#2dd4bf',
  w: '#f2b84b',
  z: '#fb923c',
  t: '#f2735b',
};

export function ParetoExplorer({ data }: Props) {
  const [stageTab, setStageTab] = useState<'l1' | 'l2'>('l1');
  const [metricTab, setMetricTab] = useState<'auc' | 'rej50'>('auc');
  const [hoveredTag, setHoveredTag] = useState<string | null>('QAT_6b_p50');

  const points: PlotPoint[] = useMemo(() => {
    if (stageTab === 'l1') {
      return data.level1_sweep.map((p) => ({
        tag: p.tag,
        bits: p.weight_bits,
        sparsityPct: Math.round(p.actual_sparsity * 1000) / 10,
        bops: p.effective_bops,
        bopReductionPct: p.bop_reduction_pct,
        acc: p.accuracy * 100,
        auc: p.macro_auc,
        rej50: p.mean_rej_50,
      }));
    }
    return data.level2_sweep.results.map((p) => ({
      tag: p.tag,
      bits: p.total_bits,
      sparsityPct: Math.round(p.actual_sparsity * 1000) / 10,
      bops: p.pquant_ebops,
      bopReductionPct: p.ebops_reduction_pct,
      acc: p.accuracy * 100,
      auc: p.macro_auc,
      rej50: p.mean_rej_50,
    }));
  }, [data, stageTab]);

  // Compute global Pareto-optimal envelope (minimizing BOPs, maximizing selected metric)
  const paretoPts = useMemo(() => {
    const sorted = [...points].sort((a, b) => a.bops - b.bops);
    const out: PlotPoint[] = [];
    let bestY = -Infinity;
    for (const pt of sorted) {
      const y = metricTab === 'auc' ? pt.auc : pt.rej50;
      if (y > bestY) {
        out.push(pt);
        bestY = y;
      }
    }
    return out;
  }, [points, metricTab]);

  const activePoint =
    points.find((p) => p.tag === hoveredTag) ??
    points.find((p) => p.tag === 'QAT_6b_p75') ??
    points[0];

  // SVG coordinate projection (Log10 inverted X axis so RIGHT = SMALLER & FASTER!)
  const W = 680;
  const H = 370;
  const pad = { l: 64, r: 28, t: 24, b: 48 };
  const logVals = points.map((p) => Math.log10(Math.max(1, p.bops)));
  const minLog = Math.min(...logVals) - 0.15;
  const maxLog = Math.max(...logVals) + 0.15;

  const yVals = points.map((p) => (metricTab === 'auc' ? p.auc : p.rej50));
  const minY = Math.min(...yVals) - (metricTab === 'auc' ? 0.008 : 15);
  const maxY = Math.max(...yVals) + (metricTab === 'auc' ? 0.005 : 15);

  const projectX = (bops: number) => {
    const lv = Math.log10(Math.max(1, bops));
    // Inverted: maxLog -> left, minLog -> right
    const norm = (maxLog - lv) / (maxLog - minLog);
    return pad.l + norm * (W - pad.l - pad.r);
  };
  const projectY = (val: number) => {
    const norm = (val - minY) / (maxY - minY);
    return H - pad.b - norm * (H - pad.t - pad.b);
  };

  const paretoPath = paretoPts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${projectX(p.bops).toFixed(1)} ${projectY(metricTab === 'auc' ? p.auc : p.rej50).toFixed(1)}`)
    .join(' ');

  const bitGroups = Array.from(new Set(points.map((p) => p.bits))).sort((a, b) => b - a);
  const yTicks = Array.from({ length: 5 }, (_, i) => minY + ((maxY - minY) * i) / 4);
  const xTickExps = Array.from(
    { length: Math.ceil(maxLog) - Math.floor(minLog) + 1 },
    (_, i) => Math.floor(minLog) + i,
  ).filter((e) => e >= minLog && e <= maxLog);

  // ROC SVG helper (Log-Y FPR vs linear TPR)
  const RW = 440;
  const RH = 320;
  const rpad = { l: 52, r: 16, t: 18, b: 42 };
  const rocX = (tpr: number) => rpad.l + tpr * (RW - rpad.l - rpad.r);
  const rocY = (fpr: number) => {
    const lf = Math.log10(Math.max(1e-3, Math.min(1, fpr)));
    // -3 -> bottom, 0 -> top
    const norm = (lf + 3) / 3;
    return RH - rpad.b - norm * (RH - rpad.t - rpad.b);
  };

  return (
    <section id="pareto" className="relative py-24 sm:py-32 detector-grid border-t border-cyan-500/10">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 mb-10">
          <div className="max-w-2xl">
            <p className="font-mono-tech text-xs tracking-[0.22em] uppercase text-amber-300">
              03 · Pareto Frontier &amp; Trigger Metrics
            </p>
            <h2 className="mt-3 font-serif-display text-4xl sm:text-6xl text-white leading-[1.02]">
              Every bit of precision, <span className="italic text-amber-300">measured</span>.
            </h2>
            <p className="mt-3 text-slate-400 text-sm sm:text-base">
              Hover any operating point to inspect the exact tradeoff between hardware compute
              (log-scale BOPs, moving right = faster silicon) and CERN trigger physics quality.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            <div className="inline-flex rounded-xl bg-[#070b14] p-1 border border-slate-800">
              <button
                onClick={() => {
                  setStageTab('l1');
                  setHoveredTag('QAT_6b_p50');
                }}
                className={`px-3.5 py-2 rounded-lg font-mono-tech text-xs transition cursor-pointer ${
                  stageTab === 'l1' ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-400/40' : 'text-slate-400 hover:text-white'
                }`}
              >
                Level 1 · 16-Feature MLP
              </button>
              <button
                onClick={() => {
                  setStageTab('l2');
                  setHoveredTag('PQuant_6b_p50');
                }}
                className={`px-3.5 py-2 rounded-lg font-mono-tech text-xs transition cursor-pointer ${
                  stageTab === 'l2' ? 'bg-amber-500/20 text-amber-200 border border-amber-400/40' : 'text-slate-400 hover:text-white'
                }`}
              >
                Level 2 · Deep Sets + PQuantML
              </button>
            </div>

            <div className="inline-flex rounded-xl bg-[#070b14] p-1 border border-slate-800">
              <button
                onClick={() => setMetricTab('auc')}
                className={`px-3 py-2 rounded-lg font-mono-tech text-xs transition cursor-pointer ${
                  metricTab === 'auc' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Macro AUC
              </button>
              <button
                onClick={() => setMetricTab('rej50')}
                className={`px-3 py-2 rounded-lg font-mono-tech text-xs transition cursor-pointer ${
                  metricTab === 'rej50' ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                1/FPR @ 50%
              </button>
            </div>
          </div>
        </div>

        {/* Key Research Callout Banner */}
        <div className="glass-panel-gold rounded-2xl p-4 sm:p-5 mb-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <Sparkles className="h-5 w-5 text-amber-300 shrink-0 mt-0.5" />
            <div>
              <div className="font-mono-tech text-[11px] uppercase tracking-widest text-amber-300 font-semibold">
                Key Pareto Discovery
              </div>
              <p className="text-sm text-slate-200 mt-0.5">
                At the same ~70k BOP budget, <strong className="text-white">6-bit + 50% pruned</strong>{' '}
                (0.9411 AUC, 253.9× rejection) strictly dominates{' '}
                <strong className="text-white">8-bit + 75% pruned</strong> (0.9388 AUC, 228.4× rejection)—proving
                that preserving neuron connectivity beats keeping the last 2 fractional bits!
              </p>
            </div>
          </div>
          {stageTab === 'l2' && data.level2_sweep.data_mode !== 'REAL_ZENODO_3602260' && (
            <span className="shrink-0 rounded-lg border border-amber-400/40 bg-amber-950/50 px-3 py-1.5 font-mono-tech text-[10px] text-amber-200">
              L2: Local Dry-Run Verified · Ready for Kaggle GPU Zenodo 3602260
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* LEFT: Interactive Log-Scale Pareto Chart */}
          <div className="lg:col-span-7 glass-panel rounded-2xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <span className="font-mono-tech text-xs text-slate-300 font-semibold">
                {stageTab === 'l1'
                  ? '12-Model Sweep · 166,000 Unseen LHC Test Jets'
                  : `20-Model CERN PQuantML Sweep · Deep Sets (N=${data.level2_sweep.n_constituents})`}
              </span>
              <div className="flex flex-wrap gap-2.5 font-mono-tech text-[10px]">
                {bitGroups.map((b) => (
                  <span key={b} className="inline-flex items-center gap-1 text-slate-300">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: BIT_COLORS[b] || '#fff' }} />
                    {b === 32 ? 'FP32' : `${b}b`}
                  </span>
                ))}
              </div>
            </div>

            <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto overflow-visible">
              {/* Grid lines */}
              {yTicks.map((yt, i) => {
                const py = projectY(yt);
                return (
                  <g key={i}>
                    <line x1={pad.l} y1={py} x2={W - pad.r} y2={py} stroke="rgba(148,163,184,0.12)" />
                    <text x={pad.l - 10} y={py + 4} textAnchor="end" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">
                      {metricTab === 'auc' ? yt.toFixed(3) : `${Math.round(yt)}×`}
                    </text>
                  </g>
                );
              })}

              {xTickExps.map((exp) => {
                const px = projectX(10 ** exp);
                return (
                  <g key={exp}>
                    <line x1={px} y1={pad.t} x2={px} y2={H - pad.b} stroke="rgba(148,163,184,0.1)" />
                    <text x={px} y={H - pad.b + 18} textAnchor="middle" fill="#94a3b8" fontSize="10" fontFamily="JetBrains Mono">
                      10^{exp}
                    </text>
                  </g>
                );
              })}

              {/* Global Pareto Envelope */}
              <path d={paretoPath} fill="none" stroke="#F2B84B" strokeWidth="2" strokeDasharray="6 4" opacity="0.85" />

              {/* Per-Bit Series Connectors */}
              {bitGroups.map((b) => {
                const sub = points.filter((p) => p.bits === b).sort((a, c) => c.bops - a.bops);
                if (sub.length < 2) return null;
                const d = sub
                  .map((p, i) => `${i === 0 ? 'M' : 'L'} ${projectX(p.bops).toFixed(1)} ${projectY(metricTab === 'auc' ? p.auc : p.rej50).toFixed(1)}`)
                  .join(' ');
                return (
                  <path key={b} d={d} fill="none" stroke={BIT_COLORS[b] || '#38e8ff'} strokeWidth="1.5" opacity="0.45" />
                );
              })}

              {/* Active Crosshair */}
              {activePoint && (
                <g>
                  <line
                    x1={projectX(activePoint.bops)}
                    y1={pad.t}
                    x2={projectX(activePoint.bops)}
                    y2={H - pad.b}
                    stroke="rgba(56,232,255,0.35)"
                    strokeDasharray="3 3"
                  />
                  <line
                    x1={pad.l}
                    y1={projectY(metricTab === 'auc' ? activePoint.auc : activePoint.rej50)}
                    x2={W - pad.r}
                    y2={projectY(metricTab === 'auc' ? activePoint.auc : activePoint.rej50)}
                    stroke="rgba(56,232,255,0.35)"
                    strokeDasharray="3 3"
                  />
                </g>
              )}

              {/* Interactive Operating Points */}
              {points.map((pt) => {
                const cx = projectX(pt.bops);
                const cy = projectY(metricTab === 'auc' ? pt.auc : pt.rej50);
                const isActive = activePoint?.tag === pt.tag;
                const col = BIT_COLORS[pt.bits] || '#38e8ff';
                return (
                  <g
                    key={pt.tag}
                    className="cursor-pointer"
                    onMouseEnter={() => setHoveredTag(pt.tag)}
                    onClick={() => setHoveredTag(pt.tag)}
                  >
                    {isActive && <circle cx={cx} cy={cy} r={13} fill={col} opacity="0.22" />}
                    <circle
                      cx={cx}
                      cy={cy}
                      r={isActive ? 7.5 : 5.5}
                      fill={col}
                      stroke="#030508"
                      strokeWidth="1.8"
                    />
                  </g>
                );
              })}

              {/* Axis Labels */}
              <text x={(W + pad.l) / 2} y={H - 8} textAnchor="middle" fill="#cbd5e1" fontSize="11" fontFamily="JetBrains Mono">
                Effective Bit-Operations (log scale · moving right = smaller &amp; faster FPGA →)
              </text>
            </svg>

            {/* Hovered Point Inspector Readout */}
            {activePoint && (
              <div className="mt-4 rounded-xl bg-[#04070d] border border-cyan-500/20 p-4 grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono-tech text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 block">Operating Point</span>
                  <span className="text-white font-bold">{activePoint.tag}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">Compute (BOPs)</span>
                  <span className="text-amber-300 font-bold">
                    {Math.round(activePoint.bops).toLocaleString()} ({activePoint.bopReductionPct > 0 ? `−${activePoint.bopReductionPct.toFixed(1)}%` : 'Base'})
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">Sparsity</span>
                  <span className="text-slate-200 font-bold">{activePoint.sparsityPct.toFixed(1)}%</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">Macro AUC</span>
                  <span className="text-cyan-300 font-bold">{activePoint.auc.toFixed(4)}</span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 block">1/FPR @ 50%</span>
                  <span className="text-emerald-300 font-bold">{activePoint.rej50.toFixed(1)}×</span>
                </div>
              </div>
            )}
          </div>

          {/* RIGHT: CERN Log-FPR ROC Curve Overlay (FP32 vs 6-bit + 75% Pruned) */}
          <div className="lg:col-span-5 glass-panel rounded-2xl p-5 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono-tech text-xs text-slate-200 font-semibold">
                  CERN Trigger ROC Convention (Log-FPR)
                </span>
                <span className="font-mono-tech text-[10px] text-slate-400">
                  Solid: FP32 · Dashed: 6b+75%
                </span>
              </div>
              <p className="text-xs text-slate-400 mb-3">
                Lower curves mean stronger background rejection at the same signal efficiency. Notice
                how our <strong>−99.12% BOPs</strong> 6-bit model hugs Float32 across all 5 classes!
              </p>

              <svg viewBox={`0 0 ${RW} ${RH}`} className="w-full h-auto">
                {[1e-3, 1e-2, 1e-1, 1].map((f) => {
                  const py = rocY(f);
                  return (
                    <g key={f}>
                      <line x1={rpad.l} y1={py} x2={RW - rpad.r} y2={py} stroke="rgba(148,163,184,0.12)" />
                      <text x={rpad.l - 8} y={py + 4} textAnchor="end" fill="#94a3b8" fontSize="9.5" fontFamily="JetBrains Mono">
                        {f === 1 ? '1.0' : `1e${Math.log10(f)}`}
                      </text>
                    </g>
                  );
                })}
                {[0.5, 0.8].map((eff) => (
                  <g key={eff}>
                    <line
                      x1={rocX(eff)}
                      y1={rpad.t}
                      x2={rocX(eff)}
                      y2={RH - rpad.b}
                      stroke="rgba(242,184,75,0.35)"
                      strokeDasharray="4 4"
                    />
                    <text x={rocX(eff)} y={RH - rpad.b + 15} textAnchor="middle" fill="#f2b84b" fontSize="9.5" fontFamily="JetBrains Mono">
                      {eff * 100}% Eff
                    </text>
                  </g>
                ))}

                {Object.entries(data.roc_curves).map(([cls, obj]) => {
                  const col = ROC_COLORS[cls] || '#38e8ff';
                  const fpPath = obj.fp32
                    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${rocX(p.tpr).toFixed(1)} ${rocY(p.fpr).toFixed(1)}`)
                    .join(' ');
                  const q6Path = obj.q6
                    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${rocX(p.tpr).toFixed(1)} ${rocY(p.fpr).toFixed(1)}`)
                    .join(' ');
                  return (
                    <g key={cls}>
                      <path d={fpPath} fill="none" stroke={col} strokeWidth="2" opacity="0.92" />
                      <path d={q6Path} fill="none" stroke={col} strokeWidth="1.7" strokeDasharray="4 3" opacity="0.85" />
                    </g>
                  );
                })}

                <text x={(RW + rpad.l) / 2} y={RH - 6} textAnchor="middle" fill="#cbd5e1" fontSize="10" fontFamily="JetBrains Mono">
                  Signal Efficiency (True Positive Rate)
                </text>
              </svg>
            </div>

            {/* Compact per-class table */}
            <div className="mt-4 pt-3 border-t border-slate-800/80 grid grid-cols-5 gap-1.5 text-center font-mono-tech text-[10px]">
              {Object.entries(data.roc_curves).map(([cls, obj]) => (
                <div key={cls} className="rounded-lg bg-[#04070d] p-2 border border-slate-800/80">
                  <div className="font-bold uppercase" style={{ color: ROC_COLORS[cls] }}>
                    {cls}
                  </div>
                  <div className="text-white mt-0.5">{obj.q6_auc.toFixed(3)}</div>
                  <div className="text-slate-500 text-[9px]">{obj.fp32_rej50}×</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
