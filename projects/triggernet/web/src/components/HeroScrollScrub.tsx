import { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { ArrowDown, Cpu, Zap } from 'lucide-react';

interface StageBeat {
  threshold: number;
  timeNs: string;
  phase: string;
  title: string;
  body: string;
  accent: string;
}

interface CachedFrame {
  t: number;
  bitmap: ImageBitmap;
}

const BEATS: StageBeat[] = [
  {
    threshold: 0.0,
    timeNs: '000.0 ns',
    phase: '40 MHz BUNCH CROSSING',
    title: 'Protons collide every 25 nanoseconds.',
    body: 'Inside the High-Luminosity LHC, collisions generate ~100 TB/s of raw detector readout—far too much to ever write to disk.',
    accent: '#38E8FF',
  },
  {
    threshold: 0.27,
    timeNs: '012.5 ns',
    phase: 'DETECTOR CALORIMETRY',
    title: 'Quarks, bosons & top quarks erupt into jets.',
    body: 'Before the next bunch crosses, the Level-1 Trigger must classify whether each particle spray is rare physics or ordinary background.',
    accent: '#38E8FF',
  },
  {
    threshold: 0.56,
    timeNs: '180.0 ns',
    phase: 'HARDWARE QUANTIZATION + PRUNING',
    title: 'Shrink the neural network onto silicon.',
    body: '32-bit float networks cannot fit on an FPGA trigger chip. TriggerNet crushes compute by 99.12% with 6-bit fixed-point QAT and 75% pruning.',
    accent: '#F2B84B',
  },
  {
    threshold: 0.80,
    timeNs: '240.0 ns',
    phase: 'L1 TRIGGER · ACCEPTED',
    title: 'Microsecond decision. Bit-accurate firmware.',
    body: 'Converted directly from PyTorch into Vivado HLS C++ via hls4ml & CERN PQuantML—verified to ΔAUC = 0.00001 against fixed-point hardware.',
    accent: '#38E8FF',
  },
];

/** Draw an ImageBitmap or HTMLVideoElement covering the full canvas (CSS object-fit: cover). */
function drawCover(
  ctx: CanvasRenderingContext2D,
  source: ImageBitmap | HTMLVideoElement,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
) {
  if (!srcW || !srcH || !dstW || !dstH) return;
  const srcRatio = srcW / srcH;
  const dstRatio = dstW / dstH;
  let drawW = dstW;
  let drawH = dstH;
  let offsetX = 0;
  let offsetY = 0;

  if (dstRatio > srcRatio) {
    drawH = dstW / srcRatio;
    offsetY = (dstH - drawH) * 0.5;
  } else {
    drawW = dstH * srcRatio;
    offsetX = (dstW - drawW) * 0.5;
  }
  ctx.drawImage(source, offsetX, offsetY, drawW, drawH);
}

export function HeroScrollScrub() {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const decoderVideoRef = useRef<HTMLVideoElement>(null);
  const clockTextRef = useRef<HTMLSpanElement>(null);
  const barFillRef = useRef<HTMLDivElement>(null);
  const statusBadgeRef = useRef<HTMLSpanElement>(null);

  const framesRef = useRef<CachedFrame[]>([]);
  const durationRef = useRef<number>(4);
  const targetProgressRef = useRef<number>(0);
  const smoothProgressRef = useRef<number>(0);
  const lastDrawnIdxRef = useRef<number>(-1);
  const activeIdxRef = useRef<number>(0);
  const isHeroVisibleRef = useRef<boolean>(true);

  const [activeIdx, setActiveIdx] = useState(0);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end end'],
  });

  const titleOpacity = useTransform(scrollYProgress, [0, 0.17], [1, 0]);
  const titleY = useTransform(scrollYProgress, [0, 0.17], [0, -34]);

  // Pause hero canvas work when scrolled past hero section
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        isHeroVisibleRef.current = entry.isIntersecting;
      },
      { threshold: 0.01 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  // Pre-decode the 4s H.264 clip in ONE fast sequential forward pass into GPU ImageBitmaps!
  // Uses an in-DOM hidden <video> so Firefox/Chrome/Safari never throttle frame decoding.
  // After ~1.5s, scrolling forward & backward renders via ctx.drawImage() in 0.1ms (zero seek lag!).
  useEffect(() => {
    const video = decoderVideoRef.current;
    if (!video) return;

    let cancelled = false;
    const captured: CachedFrame[] = [];
    let lastCapturedTime = -1;
    let fallbackRaf = 0;

    const finishCapture = () => {
      video.pause();
      if (captured.length > 0) {
        captured.sort((a, b) => a.t - b.t);
        framesRef.current = captured;
        lastDrawnIdxRef.current = -1;
        if (statusBadgeRef.current) {
          statusBadgeRef.current.textContent = `SYNCED · ${captured.length} FRAMES`;
          statusBadgeRef.current.style.color = '#34d399';
        }
      }
    };

    const grabFrame = (mediaTime: number) => {
      if (cancelled || video.readyState < 2 || video.videoWidth === 0) return;
      if (mediaTime - lastCapturedTime < 0.032 && captured.length > 0) return;
      lastCapturedTime = mediaTime;

      const targetW = Math.min(1280, video.videoWidth);
      const targetH = Math.round((targetW / video.videoWidth) * video.videoHeight);

      createImageBitmap(video, {
        resizeWidth: targetW,
        resizeHeight: targetH,
        resizeQuality: 'medium',
      })
        .then((bitmap) => {
          if (cancelled) {
            bitmap.close();
            return;
          }
          captured.push({ t: mediaTime, bitmap });
          framesRef.current = captured;

          if (captured.length === 1) {
            lastDrawnIdxRef.current = -1;
          }
          if (statusBadgeRef.current && durationRef.current > 0) {
            const pct = Math.min(99, Math.round((mediaTime / durationRef.current) * 100));
            statusBadgeRef.current.textContent = `CACHING ${pct}%`;
          }
        })
        .catch(() => {});
    };

    const startSequentialCapture = () => {
      if (cancelled) return;
      durationRef.current = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 4;
      grabFrame(0);

      // Single forward pass at 2.5x speed captures ~80-100 crisp frames in ~1.5s
      video.playbackRate = 2.5;

      const anyVideo = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number }) => void) => number;
      };

      if (typeof anyVideo.requestVideoFrameCallback === 'function') {
        const onVideoFrame = (_now: number, meta: { mediaTime: number }) => {
          if (cancelled) return;
          grabFrame(meta.mediaTime);
          if (!video.ended && video.currentTime < durationRef.current - 0.04) {
            anyVideo.requestVideoFrameCallback!(onVideoFrame);
          } else {
            finishCapture();
          }
        };
        anyVideo.requestVideoFrameCallback(onVideoFrame);
      } else {
        const pollFrame = () => {
          if (cancelled) return;
          grabFrame(video.currentTime);
          if (!video.ended && video.currentTime < durationRef.current - 0.04) {
            fallbackRaf = requestAnimationFrame(pollFrame);
          } else {
            finishCapture();
          }
        };
        fallbackRaf = requestAnimationFrame(pollFrame);
      }

      video.play().catch(() => {});
    };

    if (video.readyState >= 2) {
      startSequentialCapture();
    } else {
      video.addEventListener('loadeddata', startSequentialCapture, { once: true });
    }
    video.addEventListener('ended', finishCapture, { once: true });

    return () => {
      cancelled = true;
      cancelAnimationFrame(fallbackRaf);
      video.pause();
      captured.forEach((f) => f.bitmap.close());
      framesRef.current = [];
    };
  }, []);

  // Zero-rerender scroll listener + 60/120fps smoothed canvas scrub renderer
  useEffect(() => {
    const unsub = scrollYProgress.on('change', (p) => {
      const clamped = Math.max(0, Math.min(1, p));
      targetProgressRef.current = clamped;

      if (clockTextRef.current) {
        clockTextRef.current.textContent = (clamped * 240).toFixed(1);
      }
      if (barFillRef.current) {
        barFillRef.current.style.transform = `scaleX(${Math.max(0.04, clamped)})`;
      }

      let idx = 0;
      for (let i = 0; i < BEATS.length; i++) {
        if (clamped >= BEATS[i].threshold) idx = i;
      }
      if (idx !== activeIdxRef.current) {
        activeIdxRef.current = idx;
        setActiveIdx(idx);
      }
    });

    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
      lastDrawnIdxRef.current = -1;
    };
    handleResize();
    window.addEventListener('resize', handleResize);

    let rafId = 0;
    const renderLoop = () => {
      if (isHeroVisibleRef.current) {
        const frames = framesRef.current;
        const canvas = canvasRef.current;
        if (frames.length > 0 && canvas) {
          // Smooth Apple-style spring easing toward scroll position
          const diff = targetProgressRef.current - smoothProgressRef.current;
          if (Math.abs(diff) > 0.0004) {
            smoothProgressRef.current += diff * 0.16;
          } else {
            smoothProgressRef.current = targetProgressRef.current;
          }

          // Map smooth scroll progress directly to video timestamp (zero jump while caching!)
          const targetT = smoothProgressRef.current * durationRef.current;
          let frameIdx = 0;
          for (let i = 0; i < frames.length; i++) {
            if (frames[i].t <= targetT) frameIdx = i;
            else break;
          }

          if (frameIdx !== lastDrawnIdxRef.current) {
            const ctx = canvas.getContext('2d', { alpha: false });
            const bmp = frames[frameIdx].bitmap;
            if (ctx && bmp) {
              drawCover(ctx, bmp, bmp.width, bmp.height, canvas.width, canvas.height);
              lastDrawnIdxRef.current = frameIdx;
            }
          }
        }
      }
      rafId = requestAnimationFrame(renderLoop);
    };
    rafId = requestAnimationFrame(renderLoop);

    return () => {
      unsub();
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(rafId);
    };
  }, [scrollYProgress]);

  const beat = BEATS[activeIdx];

  return (
    <section ref={containerRef} id="hero" className="relative h-[260vh] w-full bg-[#030508]">
      <div className="sticky top-0 h-screen w-full overflow-hidden flex items-center justify-center">
        {/* In-DOM hidden single-pass decoder feeding the 120 FPS ImageBitmap canvas */}
        <video
          ref={decoderVideoRef}
          src="./media/hero_collision_fpga.mp4"
          muted
          playsInline
          preload="auto"
          className="fixed -left-[9999px] top-0 w-1 h-1 opacity-0 pointer-events-none"
        />

        {/* Instant 120 FPS Scroll-Scrubbed Canvas (zero H.264 seek lag forward OR backward!) */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 h-full w-full block opacity-95"
        />

        {/* Lightweight GPU vignette & edge fades */}
        <div className="pointer-events-none absolute inset-0 video-vignette-overlay" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#030508]/75 via-transparent to-[#030508]" />

        {/* Opening Title Lockup (fades smoothly as proton beams collide on scroll) */}
        <motion.div
          style={{ opacity: titleOpacity, y: titleY }}
          className="relative z-20 max-w-4xl px-6 text-center pointer-events-none will-change-transform"
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/35 bg-[#040a14]/90 px-4 py-1.5 mb-6">
            <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="font-mono-tech text-[11px] tracking-[0.2em] text-cyan-300 uppercase">
              CERN HL-LHC · Level-1 Trigger ML · FPGA Compression
            </span>
          </div>

          <h1 className="font-serif-display text-6xl sm:text-7xl md:text-8xl tracking-tight text-white leading-[0.95] drop-shadow-[0_4px_24px_rgba(0,0,0,0.95)]">
            Trigger<span className="italic text-cyan-300">Net</span>
          </h1>
          <p className="mt-5 text-base sm:text-lg text-slate-200 max-w-2xl mx-auto leading-relaxed drop-shadow-[0_2px_12px_rgba(0,0,0,0.9)]">
            Compressing LHC particle-jet neural networks by{' '}
            <span className="text-amber-300 font-semibold font-mono-tech">99.12%</span> in
            hardware bit-operations for microsecond FPGA inference—from raw PyTorch to{' '}
            <span className="text-cyan-300 font-mono-tech">hls4ml</span> C++ firmware &amp; CERN{' '}
            <span className="text-cyan-300 font-mono-tech">PQuantML</span>.
          </p>

          <div className="mt-10 flex flex-col items-center gap-2 text-cyan-300/90">
            <span className="font-mono-tech text-[11px] uppercase tracking-[0.25em]">
              Scroll to collide protons &amp; enter silicon
            </span>
            <ArrowDown className="h-4 w-4 animate-bounce" />
          </div>
        </motion.div>

        {/* Live 25-Nanosecond LHC Telemetry HUD (Zero-rerender DOM-ref updates!) */}
        <div className="absolute top-20 right-4 sm:right-8 z-30 glass-panel rounded-xl px-4 py-3 w-64 sm:w-72">
          <div className="flex items-center justify-between text-[10px] font-mono-tech uppercase tracking-widest text-slate-400">
            <span className="flex items-center gap-1.5">
              <Zap className="h-3 w-3 text-cyan-400" />
              L1 Trigger Clock
            </span>
            <span ref={statusBadgeRef} className="text-amber-300">
              CACHING 0%
            </span>
          </div>
          <div className="mt-1 flex items-baseline justify-between">
            <span className="font-mono-tech text-2xl font-bold text-white tabular-nums">
              <span ref={clockTextRef}>0.0</span> <span className="text-sm text-cyan-400">ns</span>
            </span>
            <span className="font-mono-tech text-[10px] text-amber-300/90">
              Budget: &lt; 1,000 ns
            </span>
          </div>
          <div className="mt-2 h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
            <div
              ref={barFillRef}
              className="h-full w-full origin-left rounded-full will-change-transform"
              style={{
                transform: 'scaleX(0.04)',
                background: 'linear-gradient(90deg, #38E8FF 0%, #F2B84B 100%)',
              }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between font-mono-tech text-[10px] text-slate-400">
            <span>Bunch Spacing: 25 ns</span>
            <span style={{ color: beat.accent }}>{beat.phase}</span>
          </div>
        </div>

        {/* Bottom Scrollytelling Beat Caption Card */}
        <motion.div
          key={activeIdx}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25 }}
          className="absolute bottom-7 left-4 right-4 sm:left-8 sm:right-auto sm:max-w-xl z-30 glass-panel rounded-2xl p-5 sm:p-6"
        >
          <div className="flex items-center gap-2 font-mono-tech text-[11px] tracking-[0.18em] uppercase mb-2">
            <Cpu className="h-3.5 w-3.5" style={{ color: beat.accent }} />
            <span style={{ color: beat.accent }}>
              t = {beat.timeNs} · {beat.phase}
            </span>
          </div>
          <h2 className="font-serif-display text-2xl sm:text-3xl text-white leading-tight">
            {beat.title}
          </h2>
          <p className="mt-2 text-sm text-slate-300 leading-relaxed">{beat.body}</p>

          <div className="mt-4 flex items-center gap-2">
            {BEATS.map((b, i) => (
              <div
                key={b.phase}
                className="h-1 rounded-full transition-all duration-200"
                style={{
                  width: i === activeIdx ? '32px' : '10px',
                  backgroundColor: i === activeIdx ? beat.accent : 'rgba(148,163,184,0.3)',
                }}
              />
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
