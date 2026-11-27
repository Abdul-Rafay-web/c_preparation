# Step 7 Shipping Checklist: From Code to GitHub + Free Zenodo DOI

Follow these exact 4 steps once your Kaggle/Colab GPU run finishes to turn **TriggerNet** into a CERN-grade public research deliverable.

---

## 1. Run Real Level 2 on Free Kaggle or Google Colab GPU (~15–20 Minutes)
Because Zenodo 3602260 is **3.87 GB** (and training 20 Deep Sets models on 504,000 jets needs a GPU):
1. Upload `notebooks/TriggerNet_Level2_Kaggle_Colab_GPU.ipynb` to **Kaggle** (Accelerator: **GPU T4** or **P100**, Internet: **ON**) or **Google Colab** (Runtime → **T4 GPU**).
2. Press **Run All**. It installs `pquant-ml`, downloads Zenodo `3602260` on datacenter fiber in ~35s, trains FP32 Deep Sets + the 20-model `PQuantML` sweep (`4–16` bits × `0%–90%` pruning), and saves `triggernet_level2_results.zip`.
3. Download `triggernet_level2_results.zip` and extract it straight into `projects/triggernet/` (this drops `figures/06_level2_hero_pareto_real.png`, `results/level2_pquant_sweep_real.json`, and checkpoints right into place!).

---

## 2. Export Your 6-Page Technical Report to PDF
1. Paste the real Level 2 table from `results/level2_pquant_sweep_real.json` into Section 4.3 of `report/triggernet_technical_report.md` and embed your figures (`03_pareto_frontier.png`, `04_pytorch_vs_hls_verification.png`, `06_level2_hero_pareto_real.png`).
2. Export `report/triggernet_technical_report.md` to a crisp PDF (via VS Code "Markdown PDF", Pandoc, or Overleaf/LaTeX).

---

## 3. Publish a Clean Dedicated GitHub Repo (`TriggerNet`)
Push `projects/triggernet/` as its own clean public repository (NEVER push `data/*.npz` or `data/*.pq` — keep `.gitignore` ignoring `data/`!):
- Pin **`TriggerNet`** as Repo #1 on your GitHub profile.
- Embed `figures/03_pareto_frontier.png` and `figures/06_level2_hero_pareto_real.png` right at the top of `README.md` so any CERN supervisor sees the Pareto frontier within 3 seconds of clicking your link.

---

## 4. Mint a Free, Permanent Scientific DOI on Zenodo (5 Minutes)
1. Sign into **[zenodo.org](https://zenodo.org)** with your GitHub account.
2. Click **New Upload** → upload your exported `TriggerNet_Technical_Report.pdf` (you can also link your GitHub release!).
3. Fill in:
   - **Resource Type:** `Publication / Technical note`
   - **Title:** `TriggerNet: Hardware-Aware Compression of Particle-Jet Taggers for the HL-LHC Level-1 Trigger`
   - **Creators:** `Abdul Rafay`
   - **Keywords:** `CERN`, `LHC`, `Level-1 Trigger`, `FPGA`, `hls4ml`, `PQuantML`, `Quantization`, `Pruning`, `Deep Sets`, `Jet Tagging`
4. Click **Publish** → Zenodo immediately mints a permanent citable link like `https://doi.org/10.5281/zenodo.XXXXXXX`.
5. Put that exact **DOI link + 1 measured headline number** directly on your CERN Summer Student & openlab CV!
