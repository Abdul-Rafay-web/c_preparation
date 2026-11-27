import { useMemo, useState } from 'react';
import { Cpu, CheckCircle2, XCircle, Shuffle, Activity } from 'lucide-react';
import type { InferenceBundle, ModelKey } from '../types';
import { runJetInference } from '../lib/apFixedInference';

interface Props {
  bundle: InferenceBundle;
}

const CLASS_META: Record<string, { name: string; color: string; massHint: string; prongs: number }> = {
  g: { name: 'Gluon (g)', color: '#64748b', massHint: 'Broad QCD spray', prongs: 1 },
  q: { name: 'Light Quark (q)', color: '#2dd4bf', massHint: 'Narrow 1-prong', prongs: 1 },
  w: { name: 'W Boson (W)', color: '#f2b84b', massHint: '2-prong · ~80.4 GeV', prongs: 2 },
  z: { name: 'Z Boson (Z)', color: '#fb923c', massHint: '2-prong · ~91.2 GeV', prongs: 2 },
  t: { name: 'Top Quark (t)', color: '#f2735b', massHint: '3-prong · ~172.7 GeV', prongs: 3 },
};

const MODEL_ORDER: ModelKey[] = ['fp32', 'qat_8b_p50', 'qat_6b_p75', 'qat_4b_p50'];

export function LiveTriggerConsole({ bundle }: Props) {
  const [selectedJetIdx, setSelectedJetIdx] = useState(20); // Starts on a textbook Top Quark jet!
  const [modelKey, setModelKey] = useState<ModelKey>('qat_6b_p75');
  const [hwFixedMode, setHwFixedMode] = useState(true);

  const jet = bundle.sample_jets[selectedJetIdx];
  const model = bundle.models[modelKey];

  const { activeRes, fp32Res, maxLogitDiff } = useMemo(() => {
    const active = runJetInference(model, jet.x_scaled, hwFixedMode);
    const ref = runJetInference(bundle.models.fp32, jet.x_scaled, false);
    const ptExact = runJetInference(model, jet.x_scaled, false);
    const diff = Math.max(...active.logits.map((v, i) => Math.abs(v - ptExact.logits[i])));
    return { activeRes: active, fp32Res: ref, maxLogitDiff: diff };
  }, [bundle, jet, model, hwFixedMode]);

  const predictedClass = bundle.class_names[activeRes.predIdx];
  const isCorrect = activeRes.predIdx === jet.true_class_idx;
  // Rare physics trigger: keep W, Z, or Top jets; discard QCD quark/gluon background!
  const triggerAccepted = ['w', 'z', 't'].includes(predictedClass);

  // Deterministic particle spray dots representing the selected jet's real multiplicity & prongs
  const sprayDots = useMemo(() => {
    const meta = CLASS_META[jet.true_class];
    const count = Math.min(48, Math.max(10, Math.round(jet.multiplicity * 0.7)));
    const centers =
      meta.prongs === 3
        ? [
            { x: 0, y: -32 },
            { x: -30, y: 22 },
            { x: 30, y: 22 },
          ]
        : meta.prongs === 2
          ? [
              { x: -24, y: -10 },
              { x: 24, y: 10 },
            ]
          : [{ x: 0, y: 0 }];

    const spread = jet.true_class === 'g' ? 42 : jet.true_class === 'q' ? 20 : 17;
    return Array.from({ length: count }, (_, i) => {
      const seed = (i + 1) * 9301 + selectedJetIdx * 49297;
      const r1 = Math.abs(Math.sin(seed * 12.9898) * 43758.5453) % 1;
      const r2 = Math.abs(Math.sin(seed * 78.233) * 19341.123) % 1;
      const c = centers[i % centers.length];
      const ang = r1 * Math.PI * 2;
      const rad = Math.sqrt(r2) * spread;
      return {
        cx: 110 + c.x + Math.cos(ang) * rad,
        cy: 110 + c.y + Math.sin(ang) * rad,
        r: 2.2 + (1 - r2) * 4.5,
      };
    });
  }, [jet, selectedJetIdx]);

  const pickRandomJet = () => {
    const next = (selectedJetIdx + 1 + Math.floor(Math.random() * 5)) % bundle.sample_jets.length;
    setSelectedJetIdx(next);
  };

  return (
    <section id="simulator" className="relative py-24 sm:py-32 bg-[#04070d] border-t border-cyan-500/10">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 mb-10">
          <div className="max-w-2xl">
            <p className="font-mono-tech text-xs tracking-[0.22em] uppercase text-cyan-400">
              02 · Live In-Browser FPGA Trigger Console
            </p>
            <h2 className="mt-3 font-serif-display text-4xl sm:text-6xl text-white leading-[1.02]">
              Test real LHC collisions <span className="italic text-cyan-300">live in your browser</span>.
            </h2>
            <p className="mt-3 text-slate-400 text-sm sm:text-base">
              Zero backend server. Your browser executes the exact exported neural network weights
              and <span className="font-mono-tech text-amber-300">ap_fixed&lt;W, I&gt;</span>{' '}
              quantized arithmetic on 25 unseen CMS/LHC test jets in under a millisecond.
            </p>
          </div>
          <button
            onClick={pickRandomJet}
            className="inline-flex items-center gap-2 self-start md:self-auto rounded-xl border border-cyan-400/30 bg-cyan-950/40 px-4 py-2.5 font-mono-tech text-xs text-cyan-200 hover:bg-cyan-900/50 transition cursor-pointer"
          >
            <Shuffle className="h-3.5 w-3.5" />
            Next Collision Event
          </button>
        </div>

        {/* Real LHC Test Event Picker Bar */}
        <div className="mb-6 flex flex-wrap gap-2">
          {bundle.sample_jets.map((j, idx) => {
            const active = idx === selectedJetIdx;
            const meta = CLASS_META[j.true_class];
            return (
              <button
                key={j.id}
                onClick={() => setSelectedJetIdx(idx)}
                className={`rounded-lg px-2.5 py-1.5 font-mono-tech text-[11px] border transition cursor-pointer ${
                  active
                    ? 'bg-cyan-500/20 border-cyan-400 text-white shadow-[0_0_16px_rgba(56,232,255,0.25)]'
                    : 'bg-[#080d17] border-slate-800 text-slate-400 hover:border-slate-600 hover:text-slate-200'
                }`}
              >
                <span
                  className="inline-block h-2 w-2 rounded-full mr-1.5"
                  style={{ backgroundColor: meta.color }}
                />
                {j.id}
                {j.difficulty === 'Borderline' && (
                  <span className="ml-1 text-[9px] text-amber-400/90">•</span>
                )}
              </button>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* LEFT: Calorimeter Event Display & Real Jet Kinematics */}
          <div className="lg:col-span-5 glass-panel rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <span className="font-mono-tech text-[10px] uppercase tracking-widest text-slate-400">
                    CMS / LHC Test Event · {jet.difficulty}
                  </span>
                  <h3 className="font-mono-tech text-xl font-bold text-white">{jet.id}</h3>
                </div>
                <span
                  className="rounded-full px-3 py-1 font-mono-tech text-xs font-semibold border"
                  style={{
                    color: CLASS_META[jet.true_class].color,
                    borderColor: `${CLASS_META[jet.true_class].color}66`,
                    backgroundColor: `${CLASS_META[jet.true_class].color}18`,
                  }}
                >
                  Truth: {jet.true_label}
                </span>
              </div>

              {/* Calorimeter (Δη, Δφ) Event Radar */}
              <div className="relative mx-auto w-full max-w-[260px] aspect-square flex items-center justify-center">
                <svg viewBox="0 0 220 220" className="w-full h-full">
                  <circle cx="110" cy="110" r="95" fill="#03060c" stroke="rgba(56,232,255,0.18)" strokeDasharray="3 4" />
                  <circle cx="110" cy="110" r="62" fill="none" stroke="rgba(56,232,255,0.12)" />
                  <circle cx="110" cy="110" r="30" fill="none" stroke="rgba(56,232,255,0.12)" />
                  <line x1="15" y1="110" x2="205" y2="110" stroke="rgba(148,163,184,0.15)" />
                  <line x1="110" y1="15" x2="110" y2="205" stroke="rgba(148,163,184,0.15)" />
                  {sprayDots.map((d, i) => (
                    <circle
                      key={i}
                      cx={d.cx}
                      cy={d.cy}
                      r={d.r}
                      fill={CLASS_META[jet.true_class].color}
                      fillOpacity={0.78}
                      stroke="#030508"
                      strokeWidth="0.6"
                    />
                  ))}
                  <text x="114" y="25" fill="rgba(148,163,184,0.7)" fontSize="9" fontFamily="JetBrains Mono">
                    +Δφ
                  </text>
                  <text x="184" y="105" fill="rgba(148,163,184,0.7)" fontSize="9" fontFamily="JetBrains Mono">
                    +Δη
                  </text>
                </svg>
              </div>
            </div>

            {/* Real Groomed Mass Gauge with W, Z, Top markers */}
            <div className="mt-4 pt-4 border-t border-slate-800/80 space-y-3">
              <div>
                <div className="flex justify-between font-mono-tech text-xs mb-1">
                  <span className="text-slate-400">Groomed Jet Mass (mass_mmdt)</span>
                  <span className="text-white font-bold">{jet.mass_gev.toFixed(1)} GeV</span>
                </div>
                <div className="relative h-2.5 w-full rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${Math.min(100, (jet.mass_gev / 240) * 100)}%`,
                      backgroundColor: CLASS_META[jet.true_class].color,
                    }}
                  />
                  {/* W (80.4), Z (91.2), Top (172.7) reference ticks */}
                  <div className="absolute top-0 bottom-0 w-[1.5px] bg-amber-300/80" style={{ left: `${(80.4 / 240) * 100}%` }} title="W: 80.4 GeV" />
                  <div className="absolute top-0 bottom-0 w-[1.5px] bg-orange-400/80" style={{ left: `${(91.2 / 240) * 100}%` }} title="Z: 91.2 GeV" />
                  <div className="absolute top-0 bottom-0 w-[1.5px] bg-rose-400/80" style={{ left: `${(172.7 / 240) * 100}%` }} title="Top: 172.7 GeV" />
                </div>
                <div className="flex justify-between font-mono-tech text-[9px] text-slate-500 mt-1">
                  <span>0 GeV</span>
                  <span className="text-amber-300/80">W 80 / Z 91</span>
                  <span className="text-rose-400/80">Top 173</span>
                  <span>240 GeV</span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 font-mono-tech text-xs">
                <div className="rounded-lg bg-[#03060c] p-2.5 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">Multiplicity</div>
                  <div className="text-white font-semibold mt-0.5">{jet.multiplicity} tracks</div>
                </div>
                <div className="rounded-lg bg-[#03060c] p-2.5 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">ECF C2(β=1)</div>
                  <div className="text-white font-semibold mt-0.5">{jet.c2_b1}</div>
                </div>
                <div className="rounded-lg bg-[#03060c] p-2.5 border border-slate-800/80">
                  <div className="text-[10px] text-slate-400">ECF D2(β=1)</div>
                  <div className="text-white font-semibold mt-0.5">{jet.d2_b1}</div>
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT: Live Hardware Precision Selector & Inference Output */}
          <div className="lg:col-span-7 glass-panel rounded-2xl p-6 flex flex-col justify-between">
            <div>
              {/* Model Compression Selector */}
              <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                <span className="font-mono-tech text-[11px] uppercase tracking-widest text-cyan-300 flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5" /> Select FPGA Precision + Sparsity
                </span>
                <label className="inline-flex items-center gap-2 cursor-pointer font-mono-tech text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={hwFixedMode}
                    onChange={(e) => setHwFixedMode(e.target.checked)}
                    className="accent-cyan-400 h-3.5 w-3.5"
                  />
                  <span>Bit-Exact <code className="text-amber-300">ap_fixed&lt;W,I&gt;</code> Emulation</span>
                </label>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mb-6">
                {MODEL_ORDER.map((k) => {
                  const m = bundle.models[k];
                  const active = k === modelKey;
                  return (
                    <button
                      key={k}
                      onClick={() => setModelKey(k)}
                      className={`rounded-xl p-3 text-left border transition cursor-pointer ${
                        active
                          ? 'bg-cyan-500/15 border-cyan-400 text-white'
                          : 'bg-[#04070d] border-slate-800 text-slate-400 hover:border-slate-600'
                      }`}
                    >
                      <div className="font-mono-tech text-xs font-bold text-white">
                        {m.bits === 32 ? 'FP32' : `${m.bits}-bit`} · {Math.round(m.sparsity_pct)}%
                      </div>
                      <div className="font-mono-tech text-[10px] text-amber-300 mt-0.5">
                        {m.bop_reduction_pct > 0 ? `−${m.bop_reduction_pct}% BOPs` : 'Baseline'}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Trigger Decision Banner */}
              <div
                className={`rounded-xl p-4 mb-6 border flex flex-wrap items-center justify-between gap-3 ${
                  triggerAccepted
                    ? 'bg-emerald-950/35 border-emerald-400/40'
                    : 'bg-slate-900/60 border-slate-700/70'
                }`}
              >
                <div className="flex items-center gap-3">
                  {isCorrect ? (
                    <CheckCircle2 className="h-6 w-6 text-emerald-400 shrink-0" />
                  ) : (
                    <XCircle className="h-6 w-6 text-rose-400 shrink-0" />
                  )}
                  <div>
                    <div className="font-mono-tech text-[10px] uppercase tracking-widest text-slate-400">
                      Predicted Particle · {isCorrect ? 'MATCHES TRUTH' : 'MISCLASSIFIED (BORDERLINE)'}
                    </div>
                    <div className="text-lg font-bold text-white">
                      {CLASS_META[predictedClass].name}{' '}
                      <span className="font-mono-tech text-sm text-cyan-300">
                        ({(activeRes.probs[activeRes.predIdx] * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                </div>

                <div
                  className={`rounded-lg px-3 py-1.5 font-mono-tech text-xs font-bold tracking-wider border ${
                    triggerAccepted
                      ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300'
                      : 'bg-rose-500/15 border-rose-400/50 text-rose-300'
                  }`}
                >
                  {triggerAccepted ? 'L1 TRIGGER: KEEP EVENT' : 'L1 TRIGGER: DISCARD QCD'}
                </div>
              </div>

              {/* Live 5-Class Softmax Probability Bars (with FP32 ghost reference ticks!) */}
              <div className="space-y-3">
                {bundle.class_names.map((c, idx) => {
                  const pct = activeRes.probs[idx] * 100;
                  const fpPct = fp32Res.probs[idx] * 100;
                  const meta = CLASS_META[c];
                  return (
                    <div key={c}>
                      <div className="flex justify-between font-mono-tech text-xs mb-1">
                        <span className={idx === jet.true_class_idx ? 'text-white font-bold' : 'text-slate-300'}>
                          {meta.name} {idx === jet.true_class_idx && '★'}
                        </span>
                        <span className="text-slate-300 tabular-nums">
                          <strong>{pct.toFixed(1)}%</strong>{' '}
                          <span className="text-[10px] text-slate-500">(FP32: {fpPct.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="relative h-3 w-full rounded-full bg-[#03060c] border border-slate-800/90 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-300"
                          style={{ width: `${Math.max(1, pct)}%`, backgroundColor: meta.color }}
                        />
                        {/* White hairline marker showing Float32 reference probability */}
                        <div
                          className="absolute top-0 bottom-0 w-0.5 bg-white/80"
                          style={{ left: `${Math.min(99.5, fpPct)}%` }}
                          title={`FP32 reference: ${fpPct.toFixed(1)}%`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Live Hardware Telemetry Footer */}
            <div className="mt-6 pt-4 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono-tech text-xs">
              <div>
                <span className="text-[10px] text-slate-500 block">JS Forward Pass</span>
                <span className="text-cyan-300 font-semibold flex items-center gap-1">
                  <Activity className="h-3 w-3" /> {activeRes.latencyMs.toFixed(2)} ms
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Non-Zero DSPs</span>
                <span className="text-white font-semibold">
                  {activeRes.activeMultipliers.toLocaleString()} / {activeRes.totalMultipliers.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Max |Δlogit| (HW)</span>
                <span className="text-amber-300 font-semibold">{maxLogitDiff.toFixed(4)}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Effective BOPs</span>
                <span className="text-emerald-300 font-semibold">{model.bops.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
