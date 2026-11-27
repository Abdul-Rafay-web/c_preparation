import type { ExportedModel } from '../types';

/**
 * Hardware-exact Xilinx `ap_fixed<W, I>` / `ap_ufixed<W, I>` rounding + saturation.
 * Matches `hls4ml` (`RND`, `SAT`) bit-for-bit directly in the browser!
 */
export function quantizeApFixed(
  val: number,
  totalBits: number,
  intBits: number,
  signed: boolean,
): number {
  const fracBits = totalBits - intBits;
  const scale = 2 ** fracBits;
  const minVal = signed ? -(2 ** (intBits - 1)) : 0;
  const maxVal = (signed ? 2 ** (intBits - 1) : 2 ** intBits) - 1 / scale;
  const clamped = Math.max(minVal, Math.min(maxVal, val));
  return Math.round(clamped * scale) / scale;
}

export interface InferenceResult {
  logits: number[];
  probs: number[];
  predIdx: number;
  latencyMs: number;
  activeMultipliers: number;
  totalMultipliers: number;
}

function softmax(logits: number[]): number[] {
  const maxL = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxL));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * Execute genuine forward pass on exported TriggerNet weights (`16 -> 64 -> 32 -> 32 -> 5`).
 * When `hardwareFixedPoint` is true (and model.bits < 32), enforces exact
 * `ap_fixed<16,6>` input/accumulators + `ap_ufixed<a_bits, a_int>` ReLU activations
 * matching the generated Vivado HLS C++ firmware.
 */
export function runJetInference(
  model: ExportedModel,
  input16: number[],
  hardwareFixedPoint: boolean,
): InferenceResult {
  const t0 = performance.now();
  const useFpgaFixed = hardwareFixedPoint && model.bits < 32;

  let vec = useFpgaFixed
    ? input16.map((v) => quantizeApFixed(v, 16, 6, true))
    : [...input16];

  let activeMultipliers = 0;
  let totalMultipliers = 0;

  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    activeMultipliers += layer.nonzero;
    totalMultipliers += layer.total;

    const next = new Array<number>(layer.out);
    for (let o = 0; o < layer.out; o++) {
      let acc = layer.b[o];
      const row = layer.w[o];
      for (let i = 0; i < layer.in; i++) {
        const w = row[i];
        if (w !== 0) {
          acc += vec[i] * w;
        }
      }
      if (useFpgaFixed) {
        acc = quantizeApFixed(acc, 16, 6, true);
      }
      // Hidden layers have ReLU (+ unsigned ap_ufixed quantization for QAT/FPGA models)
      if (l < model.layers.length - 1) {
        let act = acc > 0 ? acc : 0;
        if (model.bits < 32) {
          act = quantizeApFixed(act, model.a_bits, model.a_int, false);
        }
        next[o] = act;
      } else {
        next[o] = acc;
      }
    }
    vec = next;
  }

  const probs = softmax(vec);
  let predIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[predIdx]) predIdx = i;
  }

  return {
    logits: vec,
    probs,
    predIdx,
    latencyMs: Math.max(0.01, performance.now() - t0),
    activeMultipliers,
    totalMultipliers,
  };
}
