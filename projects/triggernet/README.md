# TriggerNet (Stage 1 — Level 1): FPGA-Ready LHC Jet Tagging

Compress, evaluate, and convert a PyTorch LHC jet-tagging neural network into **C++ FPGA firmware (`hls4ml`)** targeting the CERN Large Hadron Collider **Level-1 Trigger** microsecond latency budget.

## Research Question
How much hardware compute (**Effective Bit-Operations / BOPs**) can we eliminate from a 5-class LHC jet classifier (`g, q, W, Z, t`) using **Global L1 Magnitude Pruning** and **Fixed-Point Quantization-Aware Training (`ap_fixed<W, I>`)** before physics performance (**Macro ROC AUC** and **Background Rejection `1 / FPR` at 50% and 80% signal efficiency**) degrades—and does the converted **`hls4ml` C++ firmware** preserve physics accuracy bit-for-bit?

## Measured Results (166,000 Unseen LHC Test Jets)

| Model | Precision (`W / A`) | Actual Sparsity | Non-Zero Weights | Effective BOPs | BOP Reduction | Test Acc | Macro AUC | Mean `1/FPR` @ 50% Eff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | `FP32 / FP32` | 0.0% | 4,256 | 4,358,144 | 0.00% | 76.71% | 0.9431 | 284.1× |
| **Float32 Pruned 50%** | `FP32 / FP32` | 50.0% | 2,128 | 2,179,072 | −50.00% | 76.71% | 0.9428 | 276.3× |
| **QAT 8-bit + 0% Pruned** | `ap_fixed<8,2> / ap_ufixed<8,3>` | 3.1% | 4,124 | 263,936 | −93.94% | **76.76%** | **0.9432** | 283.7× |
| **QAT 8-bit + 50% Pruned** | `ap_fixed<8,2> / ap_ufixed<8,3>` | 50.0% | 2,128 | 136,192 | **−96.88%** | 76.72% | 0.9427 | 279.8× |
| **QAT 6-bit + 50% Pruned** | `ap_fixed<6,2> / ap_ufixed<6,3>` | 50.0% | 2,128 | 76,608 | **−98.24%** | 76.46% | 0.9411 | 253.9× |
| **QAT 6-bit + 75% Pruned** | `ap_fixed<6,2> / ap_ufixed<6,3>` | 75.0% | 1,064 | 38,304 | **−99.12%** | 75.93% | 0.9381 | 202.7× |
| **QAT 4-bit + 75% Pruned** | `ap_fixed<4,2> / ap_ufixed<4,3>` | 75.1% | 1,061 | 16,976 | **−99.61%** | 74.42% | 0.9290 | 138.1× |

### Key Findings
1. **Why QAT beats PTQ:** At 4-bit precision, naive Post-Training Quantization collapses to **57.52% accuracy (`0.8784` AUC)**, whereas our Straight-Through Estimator QAT recovers to **74.47% accuracy (`0.9301` AUC, `+0.0518` AUC rescue)**.
2. **Pareto Dominance Insight:** At a ~70k BOP budget, **`6-bit + 50% pruned`** (`0.9411` AUC, `253.9×` rejection) dominates **`8-bit + 75% pruned`** (`0.9388` AUC, `228.4×` rejection)—showing that retaining network connectivity matters more than the last 2 bits of precision.
3. **Bit-Accurate `hls4ml` Hardware Verification (20,000 test jets):**
   - **`8-bit + 50% Pruned`:** PyTorch QAT AUC `0.94235` vs. `ap_fixed` Hardware `0.94240` (**ΔAUC = `0.00005`**, **99.60%** Top-1 agreement).
   - **`6-bit + 75% Pruned`:** PyTorch QAT AUC `0.93713` vs. `ap_fixed` Hardware `0.93712` (**ΔAUC = `0.00001`**, **99.32%** Top-1 agreement).

## Notebooks & Code Structure
```
projects/triggernet/
├── download_data.py                          # Loads/caches hls4ml_lhc_jets_hlf (830k jets, 60/20/20)
├── build_notebooks.py                        # Reproducibly generates all 4 Level-1 notebooks
├── run_notebooks.py                          # Headlessly executes notebooks & embeds inline plots
├── src/
│   ├── models.py                             # Level 1: JetMLP, QuantizedJetMLP (STE ap_fixed QAT + pruning)
│   ├── deepsets.py                           # Level 2: Permutation-invariant DeepSetsJetTagger & ParticleSetTransformer
│   ├── level2_data.py                        # Level 2: Zenodo 3602260 HDF5 constituent loader + dry-run mode
│   ├── pquant_sweep.py                       # Level 2: CERN PQuantML (4-16 bit x 0-90% PDP pruning + eBOPs)
│   └── metrics.py                            # CERN ROC/AUC, Background Rejection 1/FPR, Effective BOPs
├── notebooks/
│   ├── 01_data_exploration.ipynb             # Step 1: Physics intuition, mass peaks, outliers, scaling
│   ├── 02_float_baseline.ipynb               # Step 2: Float32 PyTorch MLP + CERN trigger ROC/CM
│   ├── 03_pruning_and_quantization.ipynb     # Step 3: PTQ vs QAT, 12-model sweep, Pareto frontier
│   ├── 04_hls4ml_hardware_verification.ipynb # Step 4: hls4ml C++ firmware + bit-accurate verification
│   ├── 05_constituent_data_and_deepsets.ipynb# Step 5: Constituent jets, permutation invariance, Deep Sets vs SetTransformer
│   ├── 06_pquantml_sweep_and_pareto.ipynb    # Step 6: CERN PQuantML 4-16 bit x 0-90% sweep & Hero Pareto plot
│   └── TriggerNet_Level2_Kaggle_Colab_GPU.ipynb # 1-Click Cloud GPU runner for full 3.87 GB Zenodo 3602260
├── report/
│   └── triggernet_technical_report.md        # Step 7: 6-page CERN research technical report (Zenodo DOI ready)
├── requirements.txt                          # Reproducible Python dependencies
├── figures/                                  # Publication-quality plots (ROC, mass peaks, Pareto, parity)
├── checkpoints/                              # Saved Float32 + compressed QAT / PQuantML checkpoints
├── results/                                  # Machine-readable JSON metrics & sweep logs
└── hls_firmware/                             # Generated Vivado HLS C++ firmware (`triggernet_8b_p50`, `triggernet_6b_p75`)
```

## Reproduce Everything
```bash
../../.venv/bin/python download_data.py
../../.venv/bin/python run_notebooks.py
```
*(Install `sudo dnf install -y gcc-c++` on Fedora to also enable native `hls_model.compile()` shared-library execution inside Notebook 04.)*
