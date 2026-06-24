"""
Post-process mesh convergence study.
Hydrostatic compression at nu = 0.25 (kappa_v = 0, springs only).
Plot K error vs 1/N on log-log, extract convergence rate.

File columns (hydro-N*.txt):
  # ev P ex ey ez sx sy sz anad
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

data_dir = Path("data")
G_input = 4000.0
K_theory = 5 * G_input / 3  # at nu = 0.25, kappa_v = 0
FONT_SIZE = 12


def get_relaxed_data(fpath):
    """Load file, return last row per unique strain increment."""
    raw = np.loadtxt(fpath)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    ev_rounded = np.round(raw[:, 0], 8)
    _, idx = np.unique(ev_rounded, return_index=True)
    last_idx = np.append(idx[1:] - 1, len(raw) - 1)

    data = raw[last_idx]
    mask = np.abs(data[:, 0]) > 1e-8
    return data[mask]


# ---- Load all runs ----
files = sorted(data_dir.glob("hydro-N*.txt"))
results = []

for fpath in files:
    match = re.search(r"hydro-N(\d+)", fpath.stem)
    if not match:
        continue
    N = int(match.group(1))

    data = get_relaxed_data(fpath)
    if len(data) < 2:
        print(f"N = {N}: not enough data ({len(data)} points)")
        continue

    ev = data[:, 0]
    P = data[:, 1]

    ev_abs = np.abs(ev)
    P_abs = np.abs(P)
    mask_fit = ev_abs < 0.01
    if mask_fit.sum() < 2:
        print(f"N = {N}: not enough points below 1% strain")
        continue
    coeffs = np.polyfit(ev_abs[mask_fit], P_abs[mask_fit], 1)
    K_meas = coeffs[0]

    results.append({
        "N": N,
        "h": 1.0 / N,
        "K_meas": K_meas,
        "K_err": abs(K_meas - K_theory) / K_theory,
    })

results = sorted(results, key=lambda r: r["N"])

# ---- Print summary ----
print(f"K_theory = {K_theory:.1f}")
print(f"\n{'N':>4}  {'h':>8}  {'K_meas':>10}  {'K/Kth':>8}  {'err%':>8}")
print("-" * 44)
for r in results:
    print(f"{r['N']:4d}  {r['h']:8.4f}  {r['K_meas']:10.1f}  "
          f"{r['K_meas']/K_theory:8.4f}  {r['K_err']*100:8.2f}")

# ---- Convergence rate from log-log fit ----
N_arr = np.array([r["N"] for r in results])
err_arr = np.array([r["K_err"] for r in results])

mask = err_arr > 0
if mask.sum() >= 2:
    rate_coeffs = np.polyfit(np.log(N_arr[mask]), np.log(err_arr[mask]), 1)
    rate = rate_coeffs[0]
    print(f"\nConvergence rate: error ~ N^{rate:.2f}")
else:
    rate = None
    print("\nNot enough points for convergence rate")

# ==================================================================
# FIGURE: K error vs N on log-log
# ==================================================================
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(N_arr, err_arr * 100, "ko", ms=8, label="Measured")

N_line = np.linspace(N_arr.min() * 0.8, N_arr.max() * 1.2, 50)
if rate is not None:
    C = np.exp(rate_coeffs[1])
    ax.plot(N_line, C * N_line**rate * 100, "-", color="C1", lw=2,
            label=f"Fit: $N^{{{rate:.2f}}}$")

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)
ax.set_xticks(N_arr)
ax.set_xticklabels([str(n) for n in N_arr])

import matplotlib.ticker as ticker
yticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:g}" for tick in yticks], fontsize=FONT_SIZE)
ax.yaxis.set_minor_locator(ticker.NullLocator())

ax.set_xlabel("$N$", fontsize=1.2*FONT_SIZE)
ax.set_ylabel("$K_\mathrm{error}$ [%]", fontsize=1.2*FONT_SIZE)
# ax.set_title("Mesh convergence ($\\nu$ = 0.25, springs only)", fontsize=FONT_SIZE)
ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE)
ax.legend(fontsize=FONT_SIZE)

fig.tight_layout()
fig.savefig("convergence.png", dpi=150)

plt.show()
