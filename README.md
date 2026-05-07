# RVI-SAC: Relative Value Iteration for Continuous Control

Implementation of the **RVI-SAC** algorithm from:

> Hisaki & Ono, *"RVI-SAC: Relative Value Iteration for Continuous Control"*, ICML 2024

Experiments are run on the **Ant-v4** MuJoCo environment using Google Colab (T4 GPU).

---

## Notebooks

### `RVI_SAC_Ant_Only.ipynb` — Full Training Pipeline

Runs all four experiments from scratch and saves results to Google Drive.

| Figure | Description |
|--------|-------------|
| Fig 1  | Learning curves: RVI-SAC vs SAC (γ=0.99/0.999/0.97) vs ARO-DDPG |
| Fig 2a | RVI-SAC vs SAC-with-Reset vs ARO-DDPG-with-Reset |
| Fig 2b | Ablation — Delayed f(Q) update vs Reference State variants vs No Delay |
| Fig 2c | RVI-SAC (auto reset cost) vs fixed reset costs of 0/10/100/250/500 |

**Estimated runtime (T4 GPU, default settings):** ~20–30 min (50k steps, 2 seeds).  
Paper-faithful settings (1M steps, 10 seeds) take 4–8 hours.

---

### `RVI_SAC_Continue_From_2c.ipynb` — Resume & Plot

Loads previously saved `.npz` files for Figs 1, 2a, and 2b, trains only Fig 2c (~15–25 min), then plots all figures in paper format.

| Figure | Training Required | Data Source |
|--------|:-----------------:|-------------|
| Fig 1 (Ant panel) | No | `fig1_ant.npz` |
| Fig 2a | No | `fig2a_ant.npz` |
| Fig 2b | No | `fig2b_ant.npz` |
| Fig 2c | **Yes** | Trained in this notebook |

Also includes bonus cells to render and export a demo video of the best saved agent.

---

## Algorithms Implemented

**RVI-SAC** — the proposed method. Extends SAC with an average-reward formulation using Relative Value Iteration, a secondary reset-critic, and an auto-tuned reset cost.

**SAC** — baseline Soft Actor-Critic with configurable discount factor γ.

**SAC-with-Reset** — SAC augmented with a reset-critic and auto-tuned reset cost (ablation).

**ARO-DDPG / ARO-DDPG-with-Reset** — Average-Reward Off-policy DDPG baseline, with and without the reset-cost mechanism.

**Ablation variants:**
- `RVI_SAC_WITH_REFERENCE` — replaces the delayed f(Q) update with a fixed reference (state, action) pair.
- `RVI_SAC_WITH_FIXED_RESET_COST` — disables auto-tuning and uses a constant reset cost.
- `RVI_SAC_NO_DELAY` — removes the exponential moving average delay on the f(Q) update.

---

## Requirements

```
gymnasium[mujoco]
torch
numpy
matplotlib
tqdm
```

For the demo video cells:

```
imageio[ffmpeg]
pyvirtualdisplay
```

Install via:

```bash
pip install gymnasium[mujoco] torch numpy matplotlib tqdm imageio[ffmpeg] pyvirtualdisplay -q
apt-get install -y ffmpeg xvfb
```

> A **T4 GPU** is strongly recommended. Set it via *Runtime → Change runtime type → T4 GPU* before running.

---

## Google Drive Layout

All outputs are saved automatically to a Drive folder (default: `MyDrive/RVI_SAC_Ant_Results` or `MyDrive/RVI_SAC_Results`):

```
RVI_SAC_Results/
├── models/
│   ├── rvi_sac_ant.pt
│   ├── sac_with_reset_ant.pt
│   ├── aro_ddpg_with_reset_ant.pt
│   ├── rvi_sac_fixed_rc0_ant.pt
│   ├── rvi_sac_fixed_rc10_ant.pt
│   ├── rvi_sac_fixed_rc100_ant.pt
│   ├── rvi_sac_fixed_rc250_ant.pt
│   └── rvi_sac_fixed_rc500_ant.pt
├── data/
│   ├── fig1_ant.npz
│   ├── fig2a_ant.npz
│   ├── fig2b_ant.npz
│   └── fig2c_ant.npz
└── figures/
    ├── figure1_ant.png
    ├── figure1_ant_with_legend.png
    ├── figure2_combined.png
    ├── figure2a.png
    ├── figure2b.png
    ├── figure2c.png
    └── rvi_sac_best_demo.mp4   # (from Continue notebook)
```

---

## Key Hyperparameters

| Parameter | Default (fast) | Paper |
|-----------|:--------------:|:-----:|
| `TOTAL_STEPS` | 50,000 | 1,000,000 |
| `N_SEEDS` | 2 | 10 |
| `EVAL_INTERVAL` | 2,500 | 5,000 |
| `WARMUP_SAC` | 1,000 | 1,000 |
| `WARMUP_RVI` | 1,000 | 10,000 |
| `batch_size` | 256 | 256 |
| `lr` | 3e-4 | 3e-4 |
| `tau` (Polyak) | 0.005 | 0.005 |

---

## Recommended Run Order

1. Open `RVI_SAC_Ant_Only.ipynb` in Colab and set the runtime to T4 GPU.
2. Run all cells top-to-bottom. Results are saved to Drive after each figure.
3. If the session times out after Fig 2b, open `RVI_SAC_Continue_From_2c.ipynb` and run it to train Fig 2c and generate all final plots.

---

## Citation

```bibtex
@inproceedings{hisaki2024rvisac,
  title     = {RVI-SAC: Relative Value Iteration for Continuous Control},
  author    = {Hisaki and Ono},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```
