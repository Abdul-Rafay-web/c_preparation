# TriggerNet: Hardware-Aware Compression of Particle-Jet Taggers for the HL-LHC Level-1 Trigger

**Author:** Abdul Rafay (Software Engineering, Karachi)  
**Target Programme Context:** CERN Next Generation Triggers (NGT) / LHC Level-1 Trigger ML  
**Code Repository:** `https://github.com/Abdul-Rafay-web/TriggerNet` *(update URL after pushing)*  
**Zenodo DOI:** *(mint free DOI on zenodo.org after uploading PDF)*

---

## Abstract
At the High-Luminosity Large Hadron Collider (HL-LHC), proton bunches collide at 40 MHz, producing ~100 TB/s of detector readout that must be filtered in microseconds by Field-Programmable Gate Arrays (FPGAs) inside the Level-1 Trigger. Deploying deep neural networks within microsecond latency and strict DSP/LUT resource budgets requires co-designing compact architectures alongside fixed-point quantization and weight pruning. In this technical report, we present **TriggerNet**, a two-stage empirical study of trigger-grade LHC jet tagging across five particle classes (`g`, `q`, `W`, `Z`, `t`). In **Stage 1**, we compress a 4,389-parameter PyTorch MLP on the 830,000-jet 16-feature OpenML benchmark (`hls4ml_lhc_jets_hlf`) using global L1 magnitude pruning and Straight-Through Estimator fixed-point Quantization-Aware Training (`ap_fixed<W, I>`), achieving a **96.88% Effective Bit-Operations (BOPs) reduction** at `8-bit + 50%` pruning (`0.9427` Macro AUC vs. `0.9431` FP32 baseline; `279.8×` vs. `284.1×` mean background rejection at 50% signal efficiency) and **99.12% BOPs reduction** at `6-bit + 75%` pruning (`0.9381` AUC). Converting our compressed PyTorch models into Vivado HLS C++ firmware via `hls4ml` confirms bit-accurate parity (`ΔAUC = 0.00001`, `99.32%` Top-1 agreement at `6-bit + 75%` pruning). In **Stage 2**, we transition to raw constituent-level particle sets (`Zenodo 3602260`, up to 150 particles/jet) using a strictly permutation-invariant **Deep Sets** network (`max |Δ| < 1e-7` under random particle permutations) compressed across `4–16` bits and `0%–90%` sparsity using CERN Next Generation Triggers' **`PQuantML`** library, mapping the Pareto frontier between physics rejection and official `PQuantML` `eBOPs`.

---

## 1. Introduction: The HL-LHC Microsecond Trigger Bottleneck
Protons at the LHC collide every 25 ns (40 MHz). Because full detector readout (~100 TB/s) exceeds offline storage capacity by orders of magnitude, experiments such as ATLAS and CMS rely on a hardware **Level-1 (L1) Trigger** executed on FPGAs to accept or permanently discard collisions in O(1–10 μs). Any signal event rejected by the L1 trigger is lost forever.

Jet tagging—distinguishing sprays produced by gluons (`g`), light quarks (`q`), `W` bosons, `Z` bosons, and top quarks (`t`)—is central to discovering rare Standard Model and Beyond-Standard-Model physics. While 32-bit floating-point (`FP32`) neural networks achieve high classification accuracy, floating-point arithmetic consumes prohibitive FPGA DSP slices and latency. Quantization to arbitrary-precision fixed-point (`ap_fixed<W, I>`) and unstructured weight pruning (where exact zero weights are constant-folded away during High-Level Synthesis) dramatically reduce compute cost, measured in **Effective Bit-Operations (BOPs)**.

---

## 2. Datasets & Preprocessing

### 2.1 Stage 1: 16-Feature High-Level Jet Substructure (`hls4ml_lhc_jets_hlf`)
- **Source:** OpenML ID `42468` (830,000 simulated LHC jets; ~20% per class across `g`, `q`, `w`, `z`, `t`).
- **Features:** 16 groomed jet observables including `mass_mmdt` (revealing physical resonances at `W ≈ 80.4 GeV`, `Z ≈ 91.2 GeV`, `t ≈ 172.7 GeV`), `multiplicity`, `zlogz`, and modified Mass Drop Tagger Energy Correlation Functions (`c1`, `c2`, `d2`, `m2`, `n2`).
- **Protocol:** Stratified `60% / 20% / 20%` split (`498,000` train / `166,000` val / `166,000` test). `StandardScaler` is fit strictly on `X_train` to prevent test leakage and map inputs cleanly within `ap_fixed<16, 6>` bounds.

### 2.2 Stage 2: Constituent-Level Particle Sets (`Zenodo 3602260`)
- **Source:** CERN/FastML `hls4ml_LHCjet_150p` (`504,000` train / `126,000` val / `240,000` test jets, up to 150 particles/jet sorted by descending `pT`).
- **Representation:** Following Odagiu et al. (2024), we extract the top `N = 32` highest-`pT` constituents with 3 kinematic coordinates `(j1_ptrel, j1_etarel, j1_phirel)` per particle. Zero-padded constituent slots remain strictly `0.0` after training-only feature standardization so padding masks exclude non-existent particles cleanly.

---

## 3. Architectures, Compression & Evaluation Methodology

### 3.1 Physics-Aware Trigger Metrics
Because LHC triggers operate under extreme background dominance, we evaluate every model using:
1. **Per-Class ROC AUC** and **Macro-Average ROC AUC**.
2. **Background Rejection (`1 / FPR`)** at fixed Signal Efficiency (`TPR = 50%` and `TPR = 80%`).
3. **Hardware Compute Cost:** Effective Bit-Operations (`BOPs = non_zero_MACs × b_w × b_a`, and official `pquant.get_ebops(model)` in Stage 2).

### 3.2 Stage 1 Model & Compression
- **Architecture (`JetMLP`):** `16 → 64 → 32 → 32 → 5` (`4,256` weights, `4,389` total params, `4,358,144` FP32 BOPs).
- **Compression:** Global L1 magnitude pruning (`50%`, `75%`, `80%`) with cubic sparsity ramp + Straight-Through Estimator (STE) Quantization-Aware Training mapping weights to signed `ap_fixed<W, 2>` and ReLU activations to unsigned `ap_ufixed<W, 3>` across `8-bit`, `6-bit`, and `4-bit`.

### 3.3 Stage 2 Model & CERN `PQuantML` Pipeline
- **Architecture (`DeepSetsJetTagger`):** Permutation-invariant set network `f(X) = ρ(Mean_i(m_i · φ(x_i)))` with shared per-particle MLP `φ: 3 → 32 → 32 → 32`, masked mean pooling, and jet head `ρ: 32 → 32 → 5` (~3,461 parameters). Verified invariant to arbitrary particle shuffles (`max |f(X) - f(π(X))| < 1e-7`). Also compared against an order-free **Particle Set Transformer** (1 self-attention block without positional encoding).
- **Compression (`PQuantML`):** Wrapped via `pquant.add_compression_layers` using **Progressive Differentiable Pruning (`pdp_config`, `epsilon = 1 / prune_epochs`)** swept across `total_bits ∈ {16, 12, 8, 6, 4}` and `sparsity ∈ {0%, 50%, 75%, 90%}`.

---

## 4. Empirical Results

### 4.1 Stage 1: PTQ vs. QAT Rescue (166,000 Unseen Test Jets)
At `4-bit` fixed-point precision, naive Post-Training Quantization (PTQ) suffers catastrophic degradation, whereas QAT with STE recovers nearly full physics capability:

| Precision (`W / A`) | PTQ Acc | QAT Acc | PTQ Macro AUC | QAT Macro AUC | QAT Gain (`ΔAUC`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ap_fixed<8,2> / ap_ufixed<8,3>` | 76.70% | **76.73%** | 0.9428 | **0.9432** | +0.0004 |
| `ap_fixed<6,2> / ap_ufixed<6,3>` | 75.55% | **76.44%** | 0.9384 | **0.9413** | +0.0029 |
| `ap_fixed<4,2> / ap_ufixed<4,3>` | 57.52% | **74.47%** | 0.8784 | **0.9301** | **+0.0518** |

### 4.2 Stage 1: Pruning × Quantization Pareto Sweep (166,000 Unseen Test Jets)

| Model | Sparsity | Non-Zero Weights | Effective BOPs | BOP Reduction | Test Acc | Macro AUC | Mean `1/FPR` @ 50% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | 0.0% | 4,256 | 4,358,144 | 0.00% | 76.71% | 0.9431 | 284.1× |
| **FP32 + 50% Pruned** | 50.0% | 2,128 | 2,179,072 | −50.00% | 76.71% | 0.9428 | 276.3× |
| **QAT 8-bit + 0% Pruned** | 3.1% | 4,124 | 263,936 | −93.94% | **76.76%** | **0.9432** | 283.7× |
| **QAT 8-bit + 50% Pruned** | 50.0% | 2,128 | 136,192 | **−96.88%** | 76.72% | 0.9427 | 279.8× |
| **QAT 8-bit + 75% Pruned** | 75.0% | 1,064 | 68,096 | −98.44% | 76.01% | 0.9388 | 228.4× |
| **QAT 6-bit + 50% Pruned** | 50.0% | 2,128 | 76,608 | **−98.24%** | 76.46% | 0.9411 | 253.9× |
| **QAT 6-bit + 75% Pruned** | 75.0% | 1,064 | 38,304 | **−99.12%** | 75.93% | 0.9381 | 202.7× |
| **QAT 4-bit + 75% Pruned** | 75.1% | 1,061 | 16,976 | **−99.61%** | 74.42% | 0.9290 | 138.1× |

**Key Pareto Finding:** At a comparable ~70k BOP budget, **`6-bit + 50% pruned`** (`0.9411` AUC, `253.9×` rejection) strictly dominates **`8-bit + 75% pruned`** (`0.9388` AUC, `228.4×` rejection), demonstrating that preserving topological connectivity in a compact trigger MLP is more valuable than retaining the final 2 fractional weight bits.

*(Include Figure: `figures/03_pareto_frontier.png`)*

### 4.3 Stage 2: Constituent-Level Deep Sets & `PQuantML` Sweep (Zenodo 3602260)
> **[POPULATE AFTER RUNNING `TriggerNet_Level2_Kaggle_Colab_GPU.ipynb` ON KAGGLE/COLAB T4 GPU]**
> Replace this block with the exact JSON table from `results/level2_pquant_sweep_real.json` and embed `figures/06_level2_hero_pareto_real.png`, comparing directly against Odagiu et al. (2024) (`N=32` FP32 Deep Sets: `~73.5%` Acc, `~0.918` Macro AUC).

---

## 5. `hls4ml` FPGA Firmware Conversion & Bit-Accurate Verification
We converted both `QAT_8b_p50` and `QAT_6b_p75` into Vivado HLS C++ firmware (`hls4ml v1.3.1`, `Strategy='Latency'`, `ReuseFactor=1`, `ap_fixed<16,6>` input/accumulators, `ap_fixed<W,2>` weights, `ap_ufixed<W,3>` ReLU outputs). Inspection of generated `firmware/weights/w2.txt` confirmed **74.7% exact zeros** baked directly into hardware ROM for `triggernet_6b_p75`.

Evaluating 20,000 unseen test jets against bit-accurate `ap_fixed` hardware emulation confirms tight numerical agreement:

| Compressed Firmware | PyTorch QAT AUC | `ap_fixed` Hardware AUC | `ΔAUC` | Top-1 Agreement | Mean `|Δlogit|` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`triggernet_8b_p50`** | 0.94235 | 0.94240 | **0.00005** | **99.60%** | 0.0181 |
| **`triggernet_6b_p75`** | 0.93713 | 0.93712 | **0.00001** | **99.32%** | 0.0399 |

*(Include Figure: `figures/04_pytorch_vs_hls_verification.png`)*

---

## 6. Limitations & Future Work
1. **Effective BOPs vs. Post-Synthesis Hardware Reports:** While `BOPs` / `PQuantML eBOPs` strongly correlate with FPGA LUT/DSP consumption, actual clock-cycle latency, Initiation Interval (II), and routing congestion require full AMD Vitis HLS synthesis (planned as Level 3).
2. **Heterogeneous Bit-Widths (HGQ2):** Uniform per-layer precision was swept here; allowing per-weight/per-activation learned bit-widths via `PQuantML` + `HGQ2` can push the Pareto frontier further rightward.
3. **Full 150-Constituent Scaling:** Trigger latency scales linearly with constituent count `N`; future work will ablate `N ∈ {8, 16, 32, 64, 150}` against FPGA resource ceilings.

---

## References
1. J. Duarte et al., *Fast inference of deep neural networks in FPGAs for particle physics*, JINST 13 (2018) P07027.
2. P. Odagiu et al., *Ultrafast jet classification at the HL-LHC*, Machine Learning: Science and Technology 5(3) (2024) 035017.
3. CERN Next Generation Triggers, *PQuantML: A Tool for End-to-End Hardware-Aware Model Compression*, arXiv:2603.26595 (2026).
4. M. Zaheer et al., *Deep Sets*, NeurIPS (2017).
5. M. Pierini et al., *hls4ml LHC Jet dataset (150 particles)*, Zenodo 3602260 (2020).
