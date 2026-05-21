# 🧠 MAF: Selective Multimodal Unlearning via Boundary-Based Forgetting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" />
  <img src="https://img.shields.io/badge/Task-Multimodal%20Unlearning-purple.svg" />
  <img src="https://img.shields.io/badge/Datasets-Memotion7k%20%7C%20CREMA--D%20%7C%20MELD%20%7C%20CMU--MOSEI-orange.svg" />
  <img src="https://img.shields.io/badge/Reproducibility-Scripts%20Included-brightgreen.svg" />
</p>

<p align="center">
  <b>Selective multimodal machine unlearning for text, image, audio, and video-derived feature models.</b><br>
  Boundary deformation • Projected gradient control • Hessian-guided stabilization • Shared-space decorrelation
</p>

---

## 📌 Overview

This repository contains the implementation of **MAF**, a **Multimodal Unlearning Framework** for selectively removing the influence of targeted data from multimodal fusion models without full retraining.

MAF is designed for settings where information is distributed across multiple modalities:

- 📝 text
- 🖼️ image
- 🔊 audio
- 🎭 video-derived facial action-unit features

The framework combines decision-space and representation-space mechanisms to reduce forgotten-class accuracy, preserve retained utility, and diagnose residual privacy leakage.

---

## 🖼️ Framework

<p align="center">
  <img src="docs/figures/frame.pdf" width="900" alt="MAF Framework Overview">
</p>

<p align="center">
  <i>Figure: Overview of MAF. Replace <code>docs/figures/framework.png</code> with the final framework figure from the paper.</i>
</p>

---

## ✨ Key Features

- 🧠 **Multimodal unlearning** for fusion models with text, image, audio, and feature-level visual branches.
- 🎯 **Boundary-based forgetting** to collapse confidence on forgotten samples.
- 🧭 **Projected gradient control** to reduce interference with retained data.
- 🧮 **Hessian/Fisher-guided stabilization** for smoother unlearning updates.
- 🔗 **Shared-space decorrelation** to reduce residual cross-modal leakage.
- 🕵️ **Membership inference attack evaluation** for privacy diagnostics.
- 📊 **Paper-ready result generation** for tables and summaries.
- ⚙️ **Reproducible scripts** for main experiments and CMU-MOSEI evaluations.

---

## 🗂️ Repository Structure

```text
BoundaryUnl/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── main.py
├── config.py
├── data_loader.py
├── evaluate.py
├── trainer.py
├── unlearner.py
├── losses.py
├── baselines.py
├── plots.py
│
├── models/
│   └── model definitions
│
├── scripts/
│   ├── setup_data.py
│   ├── setup_cmu_mosei.py
│   ├── setup_cmu_mosei_3modal.py
│   ├── run_all_revision_experiments.py
│   ├── run_cmu_mosei_full.py
│   ├── run_cmu_mosei_3modal.py
│   ├── run_cmu_mosei_mia_all_baselines.py
│   └── generate_paper_tables.py
│
├── results/
│   └── final_tables/
│
└── docs/
    ├── figures/
    │   ├── framework.png
    │   ├── forget_accuracy.png
    │   ├── retain_accuracy.png
    │   └── mia_comparison.png
    └── response_to_reviewers.pdf
```


---

## 📦 Installation

Create and activate a Python environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Datasets

This project supports experiments on:

| Dataset | Modalities | Task |
|---|---|---|
| 🖼️ **Memotion7k** | Text + Image | Sentiment forgetting |
| 🔊 **CREMA-D** | Audio | Emotion / speaker forgetting |
| 💬 **MELD** | Text + Audio | Speaker-level forgetting |
| 🎙️ **CMU-MOSEI** | Text + Acoustic/Proxy Features | Sentiment forgetting |
| 🎭 **CMU-MOSEI 3-Branch Stress Test** | Text + Acoustic Proxy + FACET42 | Stress-test evaluation |

Datasets are **not included** in this repository. Use the setup scripts or follow the original dataset licenses and access instructions.

Example:

```bash
python scripts/setup_data.py
python scripts/setup_cmu_mosei.py
python scripts/setup_cmu_mosei_3modal.py
```

---

## 🚀 Running Experiments

### Run all main revision experiments

```bash
python scripts/run_all_revision_experiments.py
```

### Run main CMU-MOSEI experiment

```bash
python scripts/run_cmu_mosei_full.py
```

### Run CMU-MOSEI three-branch stress test

```bash
python scripts/run_cmu_mosei_3modal.py
```

### Run MIA evaluation for all CMU-MOSEI baselines

```bash
python scripts/run_cmu_mosei_mia_all_baselines.py
```

### Generate paper tables

```bash
python scripts/generate_paper_tables.py
```

---

## 🧪 Evaluation Metrics

| Metric | Direction | Meaning |
|---|---:|---|
| 🎯 Forget Accuracy | ↓ | Accuracy on forgotten samples; lower is better |
| ✅ Retain Accuracy | ↑ | Accuracy on retained samples; higher is better |
| ⚖️ Trade-off Score | ↑ | Balance between forgetting and retained utility |
| 🕵️ MIA ASR | ↓ | Membership inference attack success rate |
| 🔗 Shared-Space Leakage | ↓ | Residual cross-modal covariance leakage |
| ⏱️ Wall-Clock Time | ↓ | Computational cost compared with full retraining |

---

## 📊 Main Results

### Result Summary

| Dataset | Main Modality Setting | Forget Accuracy ↓ | Retain Accuracy ↑ | MIA ASR ↓ | Key Observation |
|---|---|---:|---:|---:|---|
| Memotion7k | Text + Image | 0.0309 | 0.7878 | 0.4984 | Stable below-chance forgetting in supplementary stable-seed check |
| CREMA-D | Audio | 0.0733 | 0.3804 | 0.7173 | Harder privacy/utility trade-off; audio-only setting |
| MELD | Text + Audio | 0.0000 | 0.3652 | 0.6209 | Complete forgetting but lower retained utility than some baselines |
| CMU-MOSEI | Text + Acoustic/Proxy | 0.0000 | 0.5304 | 0.4466 | Complete forgetting and strongest MIA result among compared baselines |

> Note: Results should be interpreted as approximate unlearning diagnostics, not certified deletion guarantees.

---

## 📉 Visual Results

### Forget Accuracy

<p align="center">
  <img src="docs/figures/forget_accuracy.pdf" width="750" alt="Forget Accuracy">
</p>

### Retain Accuracy

<p align="center">
  <img src="docs/figures/retain_accuracy.pdf" width="750" alt="Retain Accuracy">
</p>

### Membership Inference Attack

<p align="center">
  <img src="docs/figures/privacy_mia.pdf" width="750" alt="MIA Comparison">
</p>

If these images are not yet available, place the generated paper figures inside:

```text
docs/figures/
```

Recommended names:

```text
framework.png
forget_accuracy.png
retain_accuracy.png
mia_comparison.png
tradeoff_score.png
shared_space_leakage.png
```

---

## 🧾 CMU-MOSEI Main Evaluation

| Method | Forget ↓ | Retain ↑ | Trade-off ↑ | MIA ↓ |
|---|---:|---:|---:|---:|
| Random Relabeling | 0.0000 | 0.6735 | 0.8049 | 0.7736 |
| Fast MU | 0.0000 | 0.5332 | 0.6956 | 0.4692 |
| Delete-Refine | 0.0000 | 0.6922 | 0.8181 | 0.7174 |
| **MAF (ours)** | **0.0000** | 0.5304 | 0.6931 | **0.4466** |

MAF does not maximise retained accuracy on CMU-MOSEI, but it achieves complete forgetting with the lowest MIA score among the compared baselines.

---

## 🎭 CMU-MOSEI Three-Branch Stress Test

| Method | Forget ↓ | Retain ↑ | Trade-off ↑ | Forget Criterion |
|---|---:|---:|---:|---|
| Delete-Refine | **0.000** | **0.690** | **0.816** | Yes |
| Random Relabeling | 0.000 | 0.622 | 0.767 | Yes |
| Boundary | 0.050 | 0.651 | 0.773 | Yes |
| Amnesiac | 0.131 | 0.656 | 0.748 | Yes |
| Fisher Forgetting | 0.009 | 0.490 | 0.656 | Yes |
| **MAF (ours, λr = 3.0)** | 0.331 | 0.296 | 0.410 | Yes; slightly below chance |

This stress test is intentionally reported as a limitation-oriented analysis. It shows applicability to video-derived FACET42 features, but also exposes reduced retained accuracy in a harder multimodal setting.

---

## 🧠 Method Summary

MAF optimizes a combined objective:

```math
L_{\text{total}} =
L_{BU}
+ \lambda_1 L_G
+ \lambda_2 L_H
+ \lambda_3 L_S
```

where:

- \(L_{BU}\): boundary unlearning loss
- \(L_G\): auxiliary gradient-cosine diagnostic
- \(L_H\): Hessian/Fisher-guided stabilization
- \(L_S\): shared-space decorrelation loss

The projected update \(g_f^\perp\) is the operative gradient-control mechanism used to reduce interference between forgotten and retained objectives.

---

## 🧾 Reproducibility Settings

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 4090, 24GB VRAM |
| Framework | PyTorch 2.0 with CUDA |
| Base training | Up to 30 epochs |
| Unlearning | Up to 10 epochs |
| Base LR | \(1 \times 10^{-4}\) |
| Unlearning LR | \(5 \times 10^{-5}\) |
| Seeds | 42, 123, 456 |
| Split | Stratified 70/15/15 |
| Fusion dimension | 512 |

The supplementary CMU-MOSEI three-branch stress test uses \(\lambda_r = 3.0\) after a small tuning sweep.

---



Use `.gitignore` to keep the repository clean.

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{khan2026maf,
  title   = {Selective Multimodal Unlearning via Boundary-Based Forgetting},
  author  = {Khan, Abdullah Ahmad and Kaosar, Mohammed and Laga, Hamid and Sohel, Ferdous},
  journal = {Neural Networks},
  year    = {2026},
  note    = {Under revision}
}
```

---

## 📬 Contact

```text
Abdullah Ahmad Khan
School of Information Technology
Murdoch University, Perth, Australia
```

---

## ⚠️ Disclaimer

This repository is intended for academic research on approximate multimodal machine unlearning. The reported privacy evaluations are empirical diagnostics and should not be interpreted as certified deletion guarantees.
