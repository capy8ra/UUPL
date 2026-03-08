# Replication Guide: Baseline 2 (Benavoli & Azzimonti, 2024)

This guide explains how to replicate the Baseline 2 results from the paper using
the [`prefGP`](https://github.com/benavoli/prefGP) library.

> **Recommended environment:** Google Colab (Python 3.11). Local setup is possible
> but requires manual Python version management. The steps below cover both.

---

## Option A — Google Colab (recommended)

Open `sim_baseline2.ipynb` in Google Colab and run the cells in order.
The notebook handles all installation automatically. **Do not skip the manual
edit in Step 3 below** — it applies regardless of environment.

---

## Option B — Local Setup

### Step 1: Clone and enter the prefGP repository

```bash
git clone https://github.com/benavoli/prefGP.git
cd prefGP
sudo chmod -R u+w .
```

### Step 2: Install dependencies

The `prefGP` library requires **Python 3.11**. Check your version:

```bash
python --version
```

If you are on Python 3.10 or earlier, install 3.11 and set it as default:

```bash
# Ubuntu / Debian
sudo apt-get update -y
sudo apt-get install python3.11 python3.11-distutils

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# Reinstall pip for Python 3.11
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py

# Reinstall Jupyter kernel packages
python -m pip install ipython ipython_genutils ipykernel jupyter_console prompt_toolkit httplib2 astor
```

Then install the library's own requirements:

```bash
pip install -r requirements.txt
```

---

### Step 3: Edit `abstractModel.py` (required)

> **This step is mandatory.** Skipping it will cause bugs during sampling because
> hyperparameter optimization code in `abstractModel.py` is incompatible with the
> way the model is used here (we do not optimize hyperparameters).

Open `prefGP/model/abstractModel.py` and **comment out** the following line ranges:

| Lines | What to comment out |
|-------|---------------------|
| 3 – 6 | Import statements used only by the optimizer |
| 27 – 55 | Hyperparameter optimization method body |
| 77 – 105 | Additional optimizer-related methods |

For example, prefix each line in those ranges with `#`:

```python
# line 3
# import ...
# line 4
# import ...
# ...
```

After editing, the file should still define the `AbstractModel` class and its
`__init__` method — only the optimizer-specific imports and methods are removed.

---

### Step 4: Run the simulation

All simulation code lives in `simulations/sim_baseline2.py`. Because `prefGP`
uses local relative imports (`from model.exactPreference import ...`), the script
**must be executed from inside the `prefGP` directory**:

```bash
# From inside the prefGP/ directory:
python ../simulations/sim_baseline2.py
```

Alternatively, open and run `notebooks/sim_baseline2.ipynb` from inside the
`prefGP/` directory using Jupyter:

```bash
jupyter notebook ../notebooks/sim_baseline2.ipynb
```

---

## Expected Output

The simulation runs 50 incremental preference learning iterations. Each iteration
fits an `exactPreference` GP model via MCMC sampling (`nsamples=1000, tune=125`)
and prints the Pearson correlation between the learned reward function and the
ground-truth GMM:

```
  0%|          | 0/50 ...
>>> corr: 0.XXX
>>> corr: 0.XXX
...
```

A 3D surface plot of the final learned reward function is saved as
`sim2_baseline2.pdf` in the current working directory.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ImportError` on `model.*` | Running from wrong directory | `cd prefGP` first |
| Crash during `model.sample()` | `abstractModel.py` not edited | Complete Step 3 |
| Wrong Python version error | Python < 3.11 active | Follow Step 2 |
