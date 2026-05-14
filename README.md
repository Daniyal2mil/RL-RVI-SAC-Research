# RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning

Reproduction of **RVI-SAC** (Hisaki & Ono, ICML 2024) on MuJoCo Ant-v4.  
Paper: https://arxiv.org/abs/2408.01972  
Original repo: https://github.com/yhisaki/average-reward-drl

---

## What this notebook reproduces

| Figure | What it shows |
|--------|---------------|
| Graph 1 | RVI-SAC vs SAC (γ=0.99) vs ARO-DDPG — total return on Ant-v4 |
| Graph 2 | Q-value stability: with vs without Delayed f(Q) Update (ablation) |
| Graph 3 | Automatic Reset Cost λ converging to ε_reset = 0.001 |
| Graph 4 | RVI-SAC vs SAC-with-Reset vs ARO-DDPG (same reset structure, paper Fig 2a) |

> **Note:** The paper trains for 1M steps with 10 seeds. This notebook runs 100k steps with 1 seed due to compute constraints. Early-training dynamics are consistent with paper findings.

---

## Requirements

- Google Colab (recommended) with a **T4 GPU** runtime
- Python 3.10+
- Google Drive (for saving results)

### Python packages

```bash
pip install gymnasium[mujoco] torch numpy matplotlib scipy tqdm
```

All packages are installed automatically in **Cell 2** of the notebook. No manual installation needed if running on Colab.

---

## Setup

### 1. Open in Google Colab

Upload `RVI_SAC_Training.ipynb` to [colab.research.google.com](https://colab.research.google.com) or open directly from Google Drive.

### 2. Enable GPU

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

The notebook will raise an error and refuse to run if no GPU is detected.

### 3. Mount Google Drive

Cell 3 will prompt you to authenticate and mount your Drive. Results are saved to:

```
/content/drive/MyDrive/RVI_SAC_Results/
```

---

## Reproducing results

Run all cells in order via **Runtime → Run all**.

Expected runtime on a T4 GPU:

| Agent | Steps | Approx. time |
|-------|-------|--------------|
| RVI-SAC | 100k | ~23 min |
| SAC (γ=0.99) | 100k | ~18 min |
| RVI-SAC No-Delay (ablation) | 100k | ~23 min |
| ARO-DDPG | 100k | ~14 min |
| SAC-with-Reset | 100k | ~18 min |

**Total: ~96 minutes** for all agents and all graphs.

Evaluation runs every 10,000 steps (3 episodes per checkpoint). Training logs print to stdout:

```
step=  10000 | return=   931.2 | elapsed=82s | rc=0.81
step=  20000 | return=   142.9 | elapsed=231s | rc=2.99
...
```

`rc` is the current Reset Cost λ. You should see it rise steadily as the agent learns to avoid falls.

---

## Output files

All outputs are saved to `/content/drive/MyDrive/RVI_SAC_Results/`:

```
graph1_performance.png          # Fig 1b reproduction
graph2_qvalue_stability.png     # Fig 2b reproduction (ablation)
graph3_reset_cost.png           # Reset cost convergence
graph4_criterion_comparison.png # Fig 2a reproduction
all_graphs_combined.png         # All four graphs side by side
rvi_sac_seed0_steps.npy
rvi_sac_seed0_returns.npy
sac_seed0_returns.npy
aro_seed0_returns.npy
q_delayed_log.npy
q_nodelay_log.npy
reset_cost_log.npy
summary.json                    # Final returns for all agents
```

---

## Notebook structure

| Cell(s) | Section |
|---------|---------|
| 2–3 | Setup: install packages, mount Drive |
| 5–6 | Imports, GPU check, configuration |
| 8 | Neural network modules (actor, critic, scalar holder) |
| 10 | Optimized replay buffer (circular NumPy buffer, O(1) insert) |
| 12 | RVI-SAC implementation (3 components: Critic, Actor, Reset Cost) |
| 14 | SAC baseline (γ=0.99) |
| 16 | ARO-DDPG baseline (Saxena et al., 2023) |
| 18 | Training loop and evaluation function |
| 20 | Run all agents (RVI-SAC, SAC, No-Delay ablation, ARO-DDPG) |
| 22 | Aggregate results for plotting |
| 24 | Graph 1 — performance comparison |
| 26 | Graph 2 — Q-value stability ablation |
| 28 | Graph 3 — reset cost convergence |
| 30–31 | Graph 4 — criterion comparison (SAC-with-Reset) |
| 33 | Save all results to Drive |
| 35–36 | Combined figure |
| 38 | Install video rendering dependencies |
| 38 | Record and save demo video of trained RVI-SAC agent |

---

## Key implementation details

**Delayed f(Q) Update** — instead of using the raw mini-batch estimate f(Q';B) in the Bellman target, a smoothed running value ξ is maintained:
```
ξ ← ξ + κ · (f(Q';B) − ξ)     κ = 0.005
```
This decouples noisy batch variance from the target and prevents Q-value divergence.

**Automatic Reset Cost** — λ is updated each step via:
```
J(r_cost) = −r_cost · (ξ_reset − ε_reset)     ε_reset = 0.001
```
λ rises when the agent falls too often and falls when it is too conservative.

**Optimized replay buffer** — the original implementation used a Python list with `del memory[0]` (O(n) per step). This notebook replaces it with a pre-allocated NumPy circular buffer (O(1) insert/overwrite), giving a significant speedup at 1M capacity.

---

## Citation

```bibtex
@inproceedings{hisaki2024rvisac,
  title     = {RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning},
  author    = {Hisaki, Yukinari and Ono, Isao},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/2408.01972}
}
```
