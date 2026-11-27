import { useEffect, useState } from 'react';
import { motion, useScroll } from 'framer-motion';
import { Cpu, FileText, Github, Terminal } from 'lucide-react';
import type { DashboardData, InferenceBundle } from './types';
import { HeroScrollScrub } from './components/HeroScrollScrub';
import { SynapseShatterStory } from './components/SynapseShatterStory';
import { LiveTriggerConsole } from './components/LiveTriggerConsole';
import { ParetoExplorer } from './components/ParetoExplorer';
import { Level2Showcase } from './components/Level2Showcase';
import { FirmwareXRay } from './components/FirmwareXRay';

export default function App() {
  const [bundle, setBundle] = useState<InferenceBundle | null>(null);
  const [dash, setDash] = useState<DashboardData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const { scrollYProgress } = useScroll();

  useEffect(() => {
    Promise.all([
      fetch('./data/inference_models.json').then((r) => r.json()),
      fetch('./data/dashboard_data.json').then((r) => r.json()),
    ])
      .then(([inf, d]) => {
        setBundle(inf as InferenceBundle);
        setDash(d as DashboardData);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="min-h-screen bg-[#030508] text-slate-100">
      {/* Top Scroll Progress Beam */}
      <motion.div
        style={{ scaleX: scrollYProgress }}
        className="fixed top-0 left-0 right-0 h-[2px] origin-left z-50 bg-gradient-to-r from-cyan-400 via-white to-amber-400 will-change-transform"
      />

      {/* Sleek Navigation Bar (zero backdrop-blur overhead over 1080p video) */}
      <header className="fixed top-0 left-0 right-0 z-40 border-b border-cyan-500/15 bg-[#030508]/92">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <a href="#hero" className="flex items-center gap-2.5 group">
            <div className="h-8 w-8 rounded-lg bg-cyan-500/15 border border-cyan-400/40 flex items-center justify-center group-hover:border-cyan-300 transition">
              <Cpu className="h-4 w-4 text-cyan-300" />
            </div>
            <div>
              <span className="font-serif-display text-xl tracking-wide text-white leading-none block">
                Trigger<span className="italic text-cyan-300">Net</span>
              </span>
              <span className="font-mono-tech text-[9px] uppercase tracking-[0.2em] text-slate-400">
                CERN HL-LHC · FPGA ML
              </span>
            </div>
          </a>

          <nav className="hidden md:flex items-center gap-6 font-mono-tech text-xs text-slate-300">
            <a href="#compression" className="hover:text-cyan-300 transition">01 Synapse</a>
            <a href="#simulator" className="hover:text-cyan-300 transition">02 Simulator</a>
            <a href="#pareto" className="hover:text-cyan-300 transition">03 Pareto</a>
            <a href="#level2" className="hover:text-cyan-300 transition">04 Deep Sets</a>
            <a href="#firmware" className="hover:text-cyan-300 transition">05 hls4ml</a>
          </nav>

          <div className="flex items-center gap-2">
            <a
              href="https://github.com/Abdul-Rafay-web"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/70 px-3 py-1.5 font-mono-tech text-xs text-slate-200 hover:border-cyan-400/50 hover:text-white transition"
            >
              <Github className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">GitHub</span>
            </a>
          </div>
        </div>
      </header>

      {/* Chapter 0: Scroll-Scrubbed Google Flow 1080p Hero */}
      <HeroScrollScrub />

      {/* Data-Backed Interactive Chapters */}
      {err ? (
        <div className="max-w-xl mx-auto my-20 glass-panel rounded-2xl p-6 text-rose-300 font-mono-tech text-xs">
          Failed to load exported TriggerNet weights: {err}
        </div>
      ) : !bundle || !dash ? (
        <div className="py-24 text-center font-mono-tech text-xs text-cyan-300 animate-pulse">
          Loading exported PyTorch weights &amp; real LHC test jets...
        </div>
      ) : (
        <>
          <SynapseShatterStory bundle={bundle} />
          <LiveTriggerConsole bundle={bundle} />
          <ParetoExplorer data={dash} />
          <Level2Showcase />
          <FirmwareXRay data={dash} />
        </>
      )}

      {/* Footer / Reproducibility & CERN Application Links */}
      <footer className="border-t border-cyan-500/15 bg-[#020408] py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-8">
          <div>
            <h3 className="font-serif-display text-3xl text-white">
              Trigger<span className="italic text-cyan-300">Net</span>
            </h3>
            <p className="mt-2 text-sm text-slate-400 max-w-lg">
              Built by <strong className="text-slate-200">Abdul Rafay</strong> (Software Engineering,
              Karachi) targeting the CERN Summer Student Programme &amp; CERN openlab.
              All Level-1 metrics measured on 166,000 unseen LHC jets (<code className="text-cyan-300">OpenML 42468</code>);
              Level-2 Deep Sets engineered for <code className="text-amber-300">Zenodo 3602260</code> with{' '}
              <code className="text-amber-300">PQuantML</code> &amp; <code className="text-cyan-300">hls4ml</code>.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 font-mono-tech text-xs">
            <a
              href="https://github.com/Abdul-Rafay-web"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-950/40 px-4 py-2.5 text-cyan-200 hover:bg-cyan-900/50 transition"
            >
              <Github className="h-4 w-4" /> Source &amp; Notebooks
            </a>
            <a
              href="#firmware"
              className="inline-flex items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-950/30 px-4 py-2.5 text-amber-200 hover:bg-amber-900/40 transition"
            >
              <FileText className="h-4 w-4" /> Technical Report (Zenodo)
            </a>
            <a
              href="#simulator"
              className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-2.5 text-slate-200 hover:border-slate-500 transition"
            >
              <Terminal className="h-4 w-4" /> Live ap_fixed Console
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
