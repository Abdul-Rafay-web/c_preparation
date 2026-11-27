import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import type { InferenceBundle, ModelKey } from '../types';

interface StoryStage {
  id: string;
  modelKey: ModelKey;
  badge: string;
  title: string;
  subtitle: string;
  description: string;
  bops: string;
  reduction: string;
  acc: string;
  auc: string;
  rej50: string;
  isCrash?: boolean;
  accent: string;
}

const STAGES: StoryStage[] = [
  {
    id: 'fp32',
    modelKey: 'fp32',
    badge: 'STAGE 01 · UNCOMPRESSED REFERENCE',
    title: 'Float32 Baseline',
    subtitle: '4,256 dense 32-bit floating-point multipliers',
    description:
      'Every jet passes through 16 → 64 → 32 → 32 → 5 dense connections using 32-bit IEEE floats. Great physics separation (0.9431 Macro AUC, 284× background rejection)—but 4.36M bit-operations per jet is too wide and slow for an L1 trigger FPGA.',
    bops: '4,358,144',
    reduction: '0.0%',
    acc: '76.71%',
    auc: '0.9431',
    rej50: '284.1×',
    accent: '#38E8FF',
  },
  {
    id: 'ptq4',
    modelKey: 'qat_4b_p50',
    badge: 'STAGE 02 · WHY NAIVE ROUNDING FAILS',
    title: 'Naive 4-Bit PTQ Collapse',
    subtitle: 'Post-training rounding destroys low-bit accuracy',
    description:
      'If you simply round trained Float32 weights down to ap_fixed<4,2> without training awareness, quantization noise overwhelms the classifier—accuracy plunges from 76.71% straight down to 57.52% (AUC 0.8784). Quantization-Aware Training (QAT) fixes this.',
    bops: '68,096',
    reduction: '−98.44%',
    acc: '57.52%',
    auc: '0.8784',
    rej50: 'Crash',
    isCrash: true,
    accent: '#F2735B',
  },
  {
    id: 'qat8',
    modelKey: 'qat_8b_p50',
    badge: 'STAGE 03 · SWEET SPOT #1',
    title: 'QAT 8-Bit + 50% Pruned',
    subtitle: 'Half the synapses deleted · 96.88% compute gone',
    description:
      'Using Straight-Through Estimator (STE) QAT plus global L1 magnitude pruning, 2,128 weak weights become exact 0.0 (zero DSP multipliers in HLS!) and the surviving weights lock onto ap_fixed<8,2>. Accuracy actually ticks UP to 76.72% at 32× fewer BOPs!',
    bops: '136,192',
    reduction: '−96.88%',
    acc: '76.72%',
    auc: '0.9427',
    rej50: '279.8×',
    accent: '#38E8FF',
  },
  {
    id: 'qat6',
    modelKey: 'qat_6b_p75',
    badge: 'STAGE 04 · TRIGGER SILICON TARGET',
    title: 'QAT 6-Bit + 75% Pruned',
    subtitle: '3,192 multipliers shattered · 99.12% BOPs eliminated',
    description:
      'Three-quarters of the entire network is pruned to exact zero and every surviving weight lives on a tiny 6-bit grid (ap_fixed<6,2>, step = 1/16). Compute collapses by 114× (from 4.36M down to 38,304 BOPs) while retaining 0.9381 Macro AUC and 202.7× background rejection!',
    bops: '38,304',
    reduction: '−99.12%',
    acc: '75.93%',
    auc: '0.9381',
    rej50: '202.7×',
    accent: '#F2B84B',
  },
];

interface Props {
  bundle: InferenceBundle;
}

interface ActiveEdge {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
  speed: number;
  phase: number;
}

export function SynapseShatterStory({ bundle }: Props) {
  const [activeIdx, setActiveIdx] = useState(0);
  const sectionRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const isVisibleRef = useRef(false);

  // Track which narrative card is centered
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const idx = Number((entry.target as HTMLElement).dataset.stageIdx);
            if (!Number.isNaN(idx)) setActiveIdx(idx);
          }
        }
      },
      { rootMargin: '-38% 0px -38% 0px', threshold: 0.05 },
    );

    stageRefs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  // Viewport visibility observer: pauses canvas completely when off-screen (0% idle CPU!)
  useEffect(() => {
    const sec = sectionRef.current;
    if (!sec) return;
    const visIo = new IntersectionObserver(
      ([entry]) => {
        isVisibleRef.current = entry.isIntersecting;
      },
      { threshold: 0.01 },
    );
    visIo.observe(sec);
    return () => visIo.disconnect();
  }, []);

  // Pre-render the static synapse topology ONCE per stage change, then animate only 22 lightweight pulses!
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const stage = STAGES[activeIdx];
    const model = bundle.models[stage.modelKey];
    const layerSizes = [16, 24, 18, 18, 5];

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const W = Math.max(320, Math.round(rect.width));
    const H = Math.max(260, Math.round(rect.height));

    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);

    // Offscreen cached layer so we NEVER re-stroke 1,600 lines on every frame!
    const staticCanvas = document.createElement('canvas');
    staticCanvas.width = canvas.width;
    staticCanvas.height = canvas.height;
    const sctx = staticCanvas.getContext('2d');
    if (!sctx) return;

    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const padX = 52;
    const padY = 40;
    const colStep = (W - padX * 2) / (layerSizes.length - 1);
    const coords = layerSizes.map((count, lIdx) => {
      const x = padX + lIdx * colStep;
      const usableH = H - padY * 2;
      const yStep = count > 1 ? usableH / (count - 1) : 0;
      return Array.from({ length: count }, (_, i) => ({ x, y: padY + i * yStep }));
    });

    const strongEdges: ActiveEdge[] = [];

    // 1. Batch draw ghost zero-pruned paths in a single path call
    sctx.beginPath();
    sctx.strokeStyle = 'rgba(148, 163, 184, 0.03)';
    sctx.lineWidth = 0.5;
    for (let l = 0; l < model.layers.length; l++) {
      const layerW = model.layers[l].w;
      const src = coords[l];
      const dst = coords[l + 1];
      for (let o = 0; o < dst.length; o++) {
        const row = layerW[o % layerW.length];
        for (let i = 0; i < src.length; i++) {
          if (Math.abs(row[i % row.length]) < 1e-6) {
            sctx.moveTo(src[i].x, src[i].y);
            sctx.lineTo(dst[o].x, dst[o].y);
          }
        }
      }
    }
    sctx.stroke();

    // 2. Draw surviving non-zero weights ONCE onto the cached offscreen bitmap
    for (let l = 0; l < model.layers.length; l++) {
      const layerW = model.layers[l].w;
      const src = coords[l];
      const dst = coords[l + 1];
      for (let o = 0; o < dst.length; o++) {
        const row = layerW[o % layerW.length];
        for (let i = 0; i < src.length; i++) {
          const wVal = row[i % row.length];
          if (Math.abs(wVal) < 1e-6) continue;

          const mag = Math.min(1, Math.abs(wVal) * 1.4);
          let strokeColor: string;
          let pulseColor: string;
          if (stage.isCrash) {
            strokeColor = `rgba(242, 115, 91, ${0.14 + mag * 0.42})`;
            pulseColor = '#F2735B';
          } else if (stage.modelKey === 'qat_6b_p75') {
            strokeColor =
              wVal > 0
                ? `rgba(242, 184, 75, ${0.22 + mag * 0.58})`
                : `rgba(56, 232, 255, ${0.18 + mag * 0.52})`;
            pulseColor = wVal > 0 ? '#F2B84B' : '#38E8FF';
          } else {
            strokeColor =
              wVal > 0
                ? `rgba(56, 232, 255, ${0.11 + mag * 0.42})`
                : `rgba(242, 115, 91, ${0.09 + mag * 0.34})`;
            pulseColor = wVal > 0 ? '#38E8FF' : '#F2735B';
          }

          sctx.strokeStyle = strokeColor;
          sctx.lineWidth = 0.65 + mag * 1.3;
          sctx.beginPath();
          sctx.moveTo(src[i].x, src[i].y);
          sctx.lineTo(dst[o].x, dst[o].y);
          sctx.stroke();

          if (mag > 0.35) {
            strongEdges.push({
              x1: src[i].x,
              y1: src[i].y,
              x2: dst[o].x,
              y2: dst[o].y,
              color: pulseColor,
              speed: 0.008 + (strongEdges.length % 5) * 0.003,
              phase: (strongEdges.length * 0.37) % 1,
            });
          }
        }
      }
    }

    // 3. Draw neuron nodes ONCE onto cached offscreen bitmap (no per-frame shadowBlur!)
    const colLabels = ['16 In', '64 Dense', '32 Dense', '32 Dense', '5 Jets'];
    const outSymbols = ['g', 'q', 'W', 'Z', 't'];
    coords.forEach((col, lIdx) => {
      col.forEach((pt, i) => {
        sctx.beginPath();
        sctx.arc(pt.x, pt.y, lIdx === 4 ? 5.2 : 2.8, 0, Math.PI * 2);
        sctx.fillStyle = stage.isCrash ? '#F2735B' : lIdx === 4 ? '#F2B84B' : '#38E8FF';
        sctx.fill();

        if (lIdx === 4) {
          sctx.font = '600 11px "JetBrains Mono", monospace';
          sctx.fillStyle = '#E6EEF8';
          sctx.fillText(outSymbols[i], pt.x + 12, pt.y + 4);
        }
      });

      sctx.font = '500 10px "JetBrains Mono", monospace';
      sctx.fillStyle = 'rgba(148, 163, 184, 0.85)';
      sctx.textAlign = 'center';
      sctx.fillText(colLabels[lIdx], col[0].x, H - 12);
      sctx.textAlign = 'left';
    });

    // Pick at most 24 representative strong edges to carry animated signal pulses
    const step = Math.max(1, Math.floor(strongEdges.length / 24));
    const pulseEdges = strongEdges.filter((_, i) => i % step === 0).slice(0, 24);

    // Draw initial frame immediately
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(staticCanvas, 0, 0);

    let animId = 0;
    let t = 0;
    const animate = () => {
      if (isVisibleRef.current) {
        t += 1;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(staticCanvas, 0, 0);

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        for (let i = 0; i < pulseEdges.length; i++) {
          const e = pulseEdges[i];
          const prog = (e.phase + t * e.speed) % 1;
          const px = e.x1 + (e.x2 - e.x1) * prog;
          const py = e.y1 + (e.y2 - e.y1) * prog;
          ctx.beginPath();
          ctx.arc(px, py, 2.4, 0, Math.PI * 2);
          ctx.fillStyle = e.color;
          ctx.fill();
        }
      }
      animId = requestAnimationFrame(animate);
    };

    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, [activeIdx, bundle]);

  const stage = STAGES[activeIdx];
  const activeModel = bundle.models[stage.modelKey];
  const nonzeroTotal = activeModel.layers.reduce((a, l) => a + l.nonzero, 0);

  return (
    <section
      ref={sectionRef}
      id="compression"
      className="relative py-24 sm:py-32 detector-grid border-t border-cyan-500/10"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mb-14">
          <p className="font-mono-tech text-xs tracking-[0.22em] uppercase text-cyan-400">
            01 · The Synapse Shatter
          </p>
          <h2 className="mt-3 font-serif-display text-4xl sm:text-6xl text-white leading-[1.02]">
            Watch <span className="italic text-amber-300">99.12%</span> of compute vanish while the
            physics survives.
          </h2>
          <p className="mt-4 text-slate-400 text-base leading-relaxed">
            Every glowing line below comes directly from our real trained PyTorch weight matrices on{' '}
            <span className="text-slate-200 font-medium">830,000 LHC jets</span>. Scroll through or
            click the four compression regimes to watch pruned FPGA multipliers dissolve into dark space.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          {/* Left Narrative Column */}
          <div className="lg:col-span-5 space-y-6">
            {STAGES.map((st, idx) => {
              const selected = idx === activeIdx;
              return (
                <div
                  key={st.id}
                  ref={(el) => {
                    stageRefs.current[idx] = el;
                  }}
                  data-stage-idx={idx}
                  onClick={() => setActiveIdx(idx)}
                  className={`cursor-pointer rounded-2xl p-6 transition-colors duration-200 border ${
                    selected
                      ? 'bg-[#0b1220] border-cyan-400/50 shadow-[0_0_32px_rgba(56,232,255,0.1)]'
                      : 'bg-[#070b12]/70 border-slate-800/80 opacity-65 hover:opacity-95'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span
                      className="font-mono-tech text-[10px] tracking-[0.18em] uppercase font-semibold"
                      style={{ color: st.accent }}
                    >
                      {st.badge}
                    </span>
                    <span className="font-mono-tech text-xs px-2 py-0.5 rounded bg-slate-900 border border-slate-700 text-slate-200">
                      {st.reduction} BOPs
                    </span>
                  </div>
                  <h3 className="font-serif-display text-2xl sm:text-3xl text-white">{st.title}</h3>
                  <p className="font-mono-tech text-xs text-slate-400 mt-1">{st.subtitle}</p>
                  <p className="mt-3 text-sm text-slate-300 leading-relaxed">{st.description}</p>
                </div>
              );
            })}
          </div>

          {/* Right Sticky Live Optical Weight Canvas */}
          <div className="lg:col-span-7 lg:sticky lg:top-24">
            <div className="glass-panel rounded-2xl p-5 sm:p-6 overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
                <div>
                  <span className="font-mono-tech text-[10px] uppercase tracking-[0.2em] text-slate-400">
                    Real Exported Weight Topology
                  </span>
                  <h4 className="text-lg font-semibold text-white">{stage.title}</h4>
                </div>
                <div className="flex items-center gap-4 font-mono-tech text-xs">
                  <span className="text-cyan-300">
                    Active DSPs: <strong>{stage.isCrash ? '4,256 (4b)' : nonzeroTotal.toLocaleString()}</strong> / 4,256
                  </span>
                </div>
              </div>

              <div className="relative h-[340px] sm:h-[380px] w-full">
                <canvas ref={canvasRef} className="h-full w-full block" />
              </div>

              {/* Live Animated Metrics Bar */}
              <motion.div
                key={stage.id}
                initial={{ opacity: 0.7 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2 }}
                className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-800/80"
              >
                <div className="rounded-xl bg-[#04070d] p-3 border border-slate-800/80">
                  <div className="font-mono-tech text-[10px] uppercase text-slate-400">Effective BOPs</div>
                  <div className="font-mono-tech text-lg font-bold text-white mt-0.5">{stage.bops}</div>
                </div>
                <div className="rounded-xl bg-[#04070d] p-3 border border-slate-800/80">
                  <div className="font-mono-tech text-[10px] uppercase text-slate-400">Test Acc</div>
                  <div
                    className="font-mono-tech text-lg font-bold mt-0.5"
                    style={{ color: stage.isCrash ? '#F2735B' : '#38E8FF' }}
                  >
                    {stage.acc}
                  </div>
                </div>
                <div className="rounded-xl bg-[#04070d] p-3 border border-slate-800/80">
                  <div className="font-mono-tech text-[10px] uppercase text-slate-400">Macro AUC</div>
                  <div className="font-mono-tech text-lg font-bold text-amber-300 mt-0.5">{stage.auc}</div>
                </div>
                <div className="rounded-xl bg-[#04070d] p-3 border border-slate-800/80">
                  <div className="font-mono-tech text-[10px] uppercase text-slate-400">1/FPR @ 50%</div>
                  <div className="font-mono-tech text-lg font-bold text-emerald-300 mt-0.5">{stage.rej50}</div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
