# Towards Uncertainty Unification: A Case Study for Preference Learning

**[RSS 2025]** &nbsp;|&nbsp; [Paper (arXiv)](https://arxiv.org/abs/2503.19317v2) &nbsp;|&nbsp; [Project Page](https://sites.google.com/view/uupl-rss25/home)

**Authors:** [Shaoting Peng](https://shaotingpeng.github.io/), [Haonan Chen](https://haonan16.github.io/), [Katie Driggs-Campbell](https://krdc.web.illinois.edu/)
University of Illinois Urbana-Champaign

---

## Abstract

Learning human preferences is essential for human-robot interaction, as it enables robots to adapt their behaviors to align with human expectations and goals. However, the inherent uncertainties in both human behavior and robotic systems make preference learning a challenging task. While probabilistic robotics algorithms offer uncertainty quantification, the integration of human preference uncertainty remains underexplored. To bridge this gap, we introduce **uncertainty unification** and propose a novel framework, **uncertainty-unified preference learning (UUPL)**, which enhances Gaussian Process (GP)-based preference learning by unifying human and robot uncertainties. Specifically, UUPL includes a human preference uncertainty model that improves GP posterior mean estimation, and an uncertainty-weighted Gaussian Mixture Model (GMM) that enhances GP predictive variance accuracy. Additionally, we design a user-specific calibration process to align uncertainty representations across users, ensuring consistency and reliability in the model performance. Comprehensive experiments and user studies demonstrate that UUPL achieves state-of-the-art performance in both prediction accuracy and user rating.

---

## Repository Structure

```
.
├── uupl/
│   ├── __init__.py           # Package exports
│   ├── gp_uupl.py            # UUPL Gaussian Process (our method)
│   ├── gp_baseline3.py       # Baseline 3 GP (Bıyık et al., 2024)
│   └── utils.py              # Shared math utilities
├── simulations/
│   ├── sim_uupl.py           # Simulation runner — UUPL
│   ├── sim_baseline1.py      # Simulation runner — Baseline 1 (Chu & Ghahramani, 2005)
│   ├── sim_baseline2.py      # Simulation runner — Baseline 2 (Benavoli & Azzimonti, 2024)
│   └── sim_baseline3.py      # Simulation runner — Baseline 3 (Bıyık et al., 2024)
├── notebooks/
│   ├── sim_baseline1.ipynb   # Original Colab notebook for Baseline 1
│   ├── sim_baseline2.ipynb   # Original Colab notebook for Baseline 2
│   └── README_baseline2.md   # Step-by-step replication guide for Baseline 2
├── requirements.txt
└── .gitignore
```

---

## Method Overview

UUPL extends GP-based preference learning with three components that unify human and robot uncertainty:

1. **Human Preference Uncertainty Model** — Human confidence level `l ∈ {1, 2, 3, 4}` (very confident → very uncertain) is modelled as a noise residual `r ~ N(0, (u^l)²)` added to the latent reward difference. This is integrated into the Laplace approximation, so the posterior mean scales with the user's expressed uncertainty (Eq. 7 in the paper).

2. **Uncertainty-weighted GMM** — A GMM built from all observed query points, weighted by uncertainty level, scales the GP predictive covariance. This produces interpretable variances that jointly reflect robot epistemic uncertainty and human aleatoric uncertainty (Eq. 11–13), implemented in `uupl/gp_uupl.py`.

3. **User-specific Uncertainty Calibration** — A calibration procedure maps each user's subjective "confident / uncertain" labels to concrete `u^l` values, ensuring cross-user consistency (Algorithm 1 in the paper).

---

## Installation

```bash
git clone https://github.com/<your-org>/UUPL.git
cd UUPL
pip install -r requirements.txt
```

Python 3.9+ is recommended. Baselines 1 and 2 require additional external libraries; see their respective sections below.

---

## Running the Simulations

All simulations replicate the **Tabletop Importance** task (Simulation 2 in the paper): a 2D feature space `[-5, 5]²` with a three-component GMM as the ground-truth reward function. Accuracy is the Pearson correlation between the learned GP mean and the ground-truth GMM.

### UUPL (our method)

```bash
python -m simulations.sim_uupl
```

### Baseline 3 — Bıyık et al. (2024)

```bash
python -m simulations.sim_baseline3
```

### Baseline 1 — Chu & Ghahramani (2005)

Requires the [`GPro`](https://github.com/benavoli/GPro) library:

```bash
pip install GPro
python -m simulations.sim_baseline1
```

### Baseline 2 — Benavoli & Azzimonti (2024)

Requires the [`prefGP`](https://github.com/benavoli/prefGP) library and a **manual edit** to `abstractModel.py` before running. See [`notebooks/README_baseline2.md`](notebooks/README_baseline2.md) for the full step-by-step guide.

In short:

```bash
git clone https://github.com/benavoli/prefGP.git
cd prefGP
pip install -r requirements.txt
# Edit model/abstractModel.py as described in notebooks/README_baseline2.md
python ../simulations/sim_baseline2.py
```

---

## Citation

```bibtex
@INPROCEEDINGS{PengS-RSS-25,
    AUTHOR    = {Shaoting Peng AND Haonan Chen AND Katherine Rose Driggs-Campbell},
    TITLE     = {Towards Uncertainty Unification: A Case Study for Preference Learning},
    BOOKTITLE = {Proceedings of Robotics: Science and Systems},
    YEAR      = {2025},
    ADDRESS   = {Los Angeles, CA, USA},
    MONTH     = {June},
    DOI       = {10.15607/RSS.2025.XXI.091}
    }
```
