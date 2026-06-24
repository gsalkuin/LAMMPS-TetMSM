"""
Post-process uniaxial tension beam nu sweep.
Fixed G = 4000, varying nu.
E_input = 2*G*(1+nu) varies per run.

File columns (tens-beam-N8-nu-*.txt):
  # ez sigma_z exm eym nux nuy fz_bot
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

data_dir = Path("data")
G_input = 4000.0

# ---- Load all runs ----
files = sorted(data_dir.glob("tens-beam-N8-nu-*.txt"))
results = []

for fpath in files:
    match = re.search(r"nu-([-\d.]+)", fpath.stem)
    if not match:
        continue
    nu_in = float(match.group(1))

    E_in = 2 * G_input * (1 + nu_in)
    K_in = 2 * G_input * (1 + nu_in) / (3 * (1 - 2 * nu_in)) if abs(1 - 2 * nu_in) > 1e-6 else float('inf')

    data = np.loadtxt(fpath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if len(data) < 3:
        print(f"nu = {nu_in}: not enough data ({len(data)} points)")
        continue

    ez = data[:, 0]
    sz = data[:, 1]
    ex = data[:, 2]
    ey = data[:, 3]

    # Fit E from small strain regime (|ez| < 1%)
    mask_fit = np.abs(ez) < 0.01
    if mask_fit.sum() < 2:
        print(f"nu = {nu_in}: not enough points below 1% strain")
        continue
    coeffs = np.polyfit(ez[mask_fit], sz[mask_fit], 1)
    E_meas = coeffs[0]

    # G_meas from E_meas assuming nu_in is correct
    G_meas_from_E = E_meas / (2 * (1 + nu_in))

    # Measured nu from lateral strains (small strain)
    nux = -ex[mask_fit] / ez[mask_fit]
    nuy = -ey[mask_fit] / ez[mask_fit]
    nu_meas = np.mean((nux + nuy) / 2)

    K_meas = E_meas / (3 * (1 - 2 * nu_meas)) if abs(1 - 2 * nu_meas) > 1e-6 else float('inf')

    results.append({
        "nu_in": nu_in,
        "E_in": E_in,
        "K_in": K_in,
        "E_meas": E_meas,
        "G_meas": G_meas_from_E,
        "K_meas": K_meas,
        "nu_meas": nu_meas,
        "coeffs": coeffs,
        "data": data,
    })

results = sorted(results, key=lambda r: r["nu_in"])

# ---- Print summary ----
print(f"\n{'nu_in':>6}  {'E_in':>8}  {'E_meas':>8}  {'E/Ein':>6}  "
      f"{'G/Gin':>6}  {'nu_ms':>6}  {'dnu':>7}  {'K_in':>8}  {'K_meas':>8}")
print("-" * 80)
for r in results:
    dnu = r["nu_meas"] - r["nu_in"]
    print(f"{r['nu_in']:6.3f}  {r['E_in']:8.1f}  {r['E_meas']:8.1f}  "
          f"{r['E_meas']/r['E_in']:6.3f}  {r['G_meas']/G_input:6.3f}  "
          f"{r['nu_meas']:6.3f}  {dnu:7.4f}  "
          f"{r['K_in']:8.1f}  {r['K_meas']:8.1f}")

# ==================================================================
# FIGURE 1: Stress-strain per nu with linear fits
# ==================================================================
ncols = 3
nrows = int(np.ceil(len(results) / ncols))
fig_ss, axes_ss = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows),
                                sharex=True, sharey=True)
axes_flat = axes_ss.ravel()

for idx, r in enumerate(results):
    ax = axes_flat[idx]
    d = r["data"]

    ax.plot(d[:, 0], d[:, 1], "o", ms=3, color="C0", label="Data")

    mask_fit = np.abs(d[:, 0]) < 0.01
    ez_line = np.linspace(0, d[mask_fit, 0].max(), 50)
    sz_line = np.polyval(r["coeffs"], ez_line)
    ax.plot(ez_line, sz_line, "-", color="C1", lw=2,
            label=f"E = {r['E_meas']:.0f}")

    ax.set_title(f"$\\nu_{{in}}$ = {r['nu_in']:.3f}")
    ax.legend(fontsize=7, loc="upper left")

for j in range(len(results), len(axes_flat)):
    axes_flat[j].set_visible(False)

for ax in axes_flat:
    ax.set_xlabel("$\\varepsilon_z$")
    ax.set_ylabel("$\\sigma_z$")

fig_ss.suptitle(f"Stress-strain (beam N=8, G = {G_input:.0f})", fontsize=14)
fig_ss.tight_layout()
fig_ss.savefig("beam_ss_fits.png", dpi=150)

# ==================================================================
# FIGURE 2: E and G ratios vs nu_input
# ==================================================================
fig2_fontsize = 14
fig2_ticksize = 12
fig_sum, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

nus_in = np.array([r["nu_in"] for r in results])
E_ratio = np.array([r["E_meas"] / r["E_in"] for r in results])
G_ratio = np.array([r["G_meas"] / G_input for r in results])

# ax1.plot(nus_in, E_ratio, "ko-", ms=7, label="$E_{meas}/E_{input}$")
ax1.plot(nus_in, G_ratio, "s--", ms=6, color="C0")
ax1.axhline(1.0, ls="--", color="gray", lw=0.8)
ax1.set_xlabel("$\\nu_{input}$", fontsize=fig2_fontsize)
ax1.set_ylabel("$G_{meas}/G_{input}$", fontsize=fig2_fontsize)
# ax1.set_title(f"Elastic moduli (G = {G_input:.0f})", fontsize=16)
ax1.tick_params(axis="both", which="major", labelsize=fig2_ticksize)
# ax1.legend(fontsize=fig2_fontsize)
ax1.set_xlim(0.25, 0.5)
ax1.set_ylim(0.74, 1.01)

# Right: nu_measured vs nu_input
nu_meas_arr = np.array([r["nu_meas"] for r in results])
ax2.plot(nus_in, nu_meas_arr, "ko", ms=7)
nu_line = np.linspace(nus_in.min() - 0.05, 0.50, 100)
ax2.plot(nu_line, nu_line, "--", color="gray", lw=0.8)
ax2.set_xlabel("$\\nu_{input}$", fontsize=fig2_fontsize)
ax2.set_ylabel("$\\nu_{measured}$", fontsize=fig2_fontsize)
# ax2.set_title("Poisson's ratio")
ax2.tick_params(axis="both", which="major", labelsize=fig2_ticksize)
# ax2.legend(fontsize=fig2_fontsize)
ax2.set_xlim(0.25, 0.5)

fig_sum.tight_layout()
fig_sum.savefig("beam_G_nu_nodal.png", dpi=300)

# ==================================================================
# FIGURE 3: Errors
# ==================================================================
fig_err, (ax3, ax4) = plt.subplots(1, 2, figsize=(11, 4.5))

E_err = (np.array([r["E_meas"] for r in results]) -
         np.array([r["E_in"] for r in results])) / \
         np.array([r["E_in"] for r in results]) * 100
ax3.plot(nus_in, E_err, "ko-", ms=7)
ax3.axhline(0.0, ls="--", color="gray", lw=0.8)
ax3.set_xlabel("$\\nu_{input}$")
ax3.set_ylabel("$(E_{meas} - E_{input}) / E_{input}$ [%]")
ax3.set_title("Young's modulus error")

nu_err = nu_meas_arr - nus_in
ax4.plot(nus_in, nu_err, "ko-", ms=7)
ax4.axhline(0.0, ls="--", color="gray", lw=0.8)
ax4.set_xlabel("$\\nu_{input}$")
ax4.set_ylabel("$\\nu_{meas} - \\nu_{input}$")
ax4.set_title("Poisson's ratio error")

fig_err.tight_layout()
fig_err.savefig("beam_error.png", dpi=150)

plt.show()
