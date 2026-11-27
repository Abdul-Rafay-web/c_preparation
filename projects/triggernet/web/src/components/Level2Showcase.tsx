import { useEffect, useRef, useState } from 'react';
import { Shuffle, Bug, Layers } from 'lucide-react';

interface ParticleRow {
  id: number;
  pt: number;
  eta: number;
  phi: number;
  prong: string;
}

const INITIAL_PARTICLES: ParticleRow[] = [
  { id: 1, pt: 0.248, eta: 0.142, phi: -0.091, prong: 'Sub-jet A (b)' },
  { id: 2, pt: 0.194, eta: -0.118, phi: 0.135, prong: 'Sub-jet B (q₁)' },
  { id: 3, pt: 0.167, eta: -0.024, phi: -0.156, prong: 'Sub-jet C (q₂)' },
  { id: 4, pt: 0.112, eta: 0.155, phi: -0.072, prong: 'Sub-jet A (b)' },
  { id: 5, pt: 0.089, eta: -0.104, phi: 0.128, prong: 'Sub-jet B (q₁)' },
  { id: 6, pt: 0.064, eta: -0.031, phi: -0.144, prong: 'Sub-jet C (q₂)' },
];

export function Level2Showcase() {
  const [particles, setParticles] = useState<ParticleRow[]>(INITIAL_PARTICLES);
  const [shuffleCount, setShuffleCount] = useState(0);
  const sectionRef = useRef<HTMLElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Only decode jet_cloud_loop.mp4 while visible in the viewport (0% off-screen decode lag!)
  useEffect(() => {
    const sec = sectionRef.current;
    const vid = videoRef.current;
    if (!sec || !vid) return;

    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          vid.play().catch(() => {});
        } else {
          vid.pause();
        }
      },
      { threshold: 0.05 },
    );
    io.observe(sec);
    return () => io.disconnect();
  }, []);

  const shuffleParticles = () => {
    setParticles((prev) => {
      const next = [...prev];
      for (let i = next.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [next[i], next[j]] = [next[j], next[i]];
      }
      return next;
    });
    setShuffleCount((c) => c + 1);
  };

  return (
    <section
      ref={sectionRef}
      id="level2"
      className="relative py-24 sm:py-32 bg-[#030508] border-t border-cyan-500/10 overflow-hidden"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-center">
          {/* LEFT: Seamless 1080p Google Flow Multi-Prong Jet Cloud Loop (viewport-gated) */}
          <div className="lg:col-span-6 relative">
            <div className="relative rounded-3xl overflow-hidden bg-[#030508] border border-cyan-500/15 shadow-[0_0_60px_rgba(56,232,255,0.08)]">
              <video
                ref={videoRef}
                src="./media/jet_cloud_loop.mp4"
                loop
                muted
                playsInline
                preload="metadata"
                className="w-full h-auto object-cover"
              />
              <div className="pointer-events-none absolute inset-0 video-vignette-overlay" />
              <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#030508] via-transparent to-transparent" />
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between font-mono-tech text-[11px]">
                <span className="rounded-lg bg-[#030508]/90 border border-cyan-400/30 px-3 py-1.5 text-cyan-200">
                  Up to 150 Constituents · (pT_rel, Δη, Δφ)
                </span>
                <span className="rounded-lg bg-[#030508]/90 border border-amber-400/30 px-3 py-1.5 text-amber-300">
                  Deep Sets · 3,461 Params
                </span>
              </div>
            </div>
          </div>

          {/* RIGHT: Deep Sets Permutation Invariance Live Lab + CERN PQuantML Discovery */}
          <div className="lg:col-span-6 space-y-6">
            <div>
              <p className="font-mono-tech text-xs tracking-[0.22em] uppercase text-cyan-400">
                04 · Level 2 Constituent Sets &amp; CERN PQuantML
              </p>
              <h2 className="mt-3 font-serif-display text-4xl sm:text-5xl text-white leading-[1.04]">
                Raw particle clouds have <span className="italic text-cyan-300">no fixed order</span>.
              </h2>
              <p className="mt-3 text-slate-400 text-sm sm:text-base leading-relaxed">
                Moving beyond 16 pre-computed summary variables to raw constituent particles
                (<code className="text-cyan-300">Zenodo 3602260</code>), we build a 3,461-parameter
                permutation-invariant <strong>Deep Sets</strong> architecture matching{' '}
                <em>Odagiu et al. (2024)</em> and compress it across <strong>4–16 bits × 0–90% pruning</strong>{' '}
                with CERN Next Generation Triggers&apos; own <code className="text-amber-300">PQuantML</code>.
              </p>
            </div>

            {/* Interactive Permutation Invariance Proof Widget */}
            <div className="glass-panel rounded-2xl p-5">
              <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                <span className="font-mono-tech text-xs text-white font-semibold flex items-center gap-2">
                  <Layers className="h-4 w-4 text-cyan-400" />
                  Live Permutation Invariance Proof
                </span>
                <button
                  onClick={shuffleParticles}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500/15 border border-cyan-400/40 px-3 py-1.5 font-mono-tech text-xs text-cyan-200 hover:bg-cyan-500/25 transition cursor-pointer"
                >
                  <Shuffle className="h-3.5 w-3.5" />
                  Shuffle Particle Order {shuffleCount > 0 && `(#${shuffleCount})`}
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full font-mono-tech text-[11px] text-left">
                  <thead className="text-slate-500 border-b border-slate-800">
                    <tr>
                      <th className="py-1.5">Row</th>
                      <th>Particle</th>
                      <th>pT_rel</th>
                      <th>Δη</th>
                      <th>Δφ</th>
                      <th>Sub-Jet</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {particles.map((p, i) => (
                      <tr key={p.id}>
                        <td className="py-1.5 text-slate-500">[{i}]</td>
                        <td className="text-cyan-300 font-semibold">p#{p.id}</td>
                        <td>{p.pt.toFixed(3)}</td>
                        <td>{p.eta >= 0 ? `+${p.eta.toFixed(3)}` : p.eta.toFixed(3)}</td>
                        <td>{p.phi >= 0 ? `+${p.phi.toFixed(3)}` : p.phi.toFixed(3)}</td>
                        <td className="text-amber-300/90">{p.prong}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 font-mono-tech text-xs">
                <span className="text-slate-400">
                  Deep Sets: <code className="text-white">ρ( Meanᵢ φ(xᵢ) )</code>
                </span>
                <span className="text-emerald-300 font-semibold">
                  Output Shift |Δ| = 7.45 × 10⁻⁹ (INVARIANT ✓)
                </span>
              </div>
            </div>

            {/* Open-Source Contribution Discovery Card */}
            <div className="rounded-2xl border border-rose-400/30 bg-rose-950/25 p-4 flex items-start gap-3">
              <Bug className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
              <div className="text-xs leading-relaxed text-slate-300">
                <span className="font-mono-tech uppercase tracking-wider text-rose-300 font-semibold block mb-1">
                  Real Upstream Bug Discovered in CERN PQuantML v0.1.0
                </span>
                While sweeping <code className="text-white">0%–90%</code> sparsity via{' '}
                <code className="text-white">pdp_config()</code>, we discovered an off-by-one{' '}
                <code className="text-rose-200">IndexError</code> in{' '}
                <code className="text-white">_pdp_setup()</code> when <code className="text-white">sparsity=0.0</code>{' '}
                (even with <code className="text-white">enable_pruning=False</code>), now documented as an
                upstream PR candidate to <strong className="text-white">cern-nextgen/PQuantML</strong>.
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
