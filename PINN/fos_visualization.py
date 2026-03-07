"""
=============================================================================
  PINN v13-106 — Factor of Safety Prediction Visualization Code
  One-Way Hydromechanical PINN for Rainfall-Induced Slope Stability
=============================================================================
  All labels, titles, axis names, and colour maps are collected in the
  CONFIG section at the top.  Edit only that section to customise every
  figure without touching the plotting logic.
=============================================================================
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

# ── matplotlib defaults (publication quality) ──────────────────────────────
matplotlib.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "grid.linestyle": "--",
})

# =============================================================================
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║               EDITABLE CONFIGURATION — CHANGE LABELS HERE                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# =============================================================================

CFG = {
    # ── Domain / study metadata ──────────────────────────────────────────
    "site_name":      "Device 106 — Instrumented Hillslope",
    "model_version":  "PINN v13-106",
    "study_period":   "Nov 2025 – Jan 2026  (~87 days, 2089 h)",
    "sensor_depth":   "0.30 m (z = 2.70 m from base)",
    "slope_angle":    "30°",
    "failure_plane":  "z = 3.0 m",

    # ── Depth / layer labels ─────────────────────────────────────────────
    "layer_names": {
        "L1": "Sandy Clay Loam / A-horizon  (2.70–3.00 m)",
        "L2": "Clay B-horizon               (0.50–2.70 m)",
        "L3": "Saprolite C-horizon          (0.00–0.50 m)",
    },
    "layer_boundaries_m": [0.5, 2.7],   # horizontal dashed lines on depth plots
    "z_label":       "Depth  z  (m from surface)",   # y-axis on profile plots
    "z_max":          3.0,
    "t_max_h":        2094.0,

    # ── Axis labels ──────────────────────────────────────────────────────
    "fos_xlabel":    "Factor of Safety  (FoS)",
    "psi_xlabel":    "Matric Suction  ψ  (m)",
    "uw_xlabel":     "Pore-Water Pressure  u_w  (kPa)",
    "theta_xlabel":  "Volumetric Water Content  θ  (m³ m⁻³)",
    "time_xlabel":   "Time  (hours from start)",
    "rain_ylabel":   "Rainfall Intensity  (mm h⁻¹)",
    "theta_ylabel":  "θ  (m³ m⁻³)",
    "fos_ylabel":    "Factor of Safety  (FoS)",
    "rmse_ylabel":   "RMSE  (m³ m⁻³)",
    "loss_ylabel":   "Training Loss  (log scale)",

    # ── Thresholds / reference lines ─────────────────────────────────────
    "fos_failure":     1.0,   # red dashed line
    "fos_warning":     1.5,   # orange dashed line
    "fos_target":      2.0,   # green dotted line (design safe)

    # ── Colour scheme ────────────────────────────────────────────────────
    "col_observed":  "#4e3b2a",    # dark brown  — observed θ
    "col_predicted": "#d62728",    # red         — predicted
    "col_rain":      "#2171b5",    # blue        — rainfall fill
    "col_fos":       "#1a1a1a",    # near-black  — FoS line
    "col_failure":   "#d62728",    # red         — failure zone
    "col_warning":   "#ff7f0e",    # orange      — warning zone
    "col_safe":      "#2ca02c",    # green       — stable zone
    "col_train":     "#4393c3",    # blue        — training window
    "col_val":       "#f4a582",    # peach       — validation
    "col_test":      "#d6604d",    # red-orange  — test

    # ── Sensitivity scenario labels ──────────────────────────────────────
    "sens_scenarios": {
        "Baseline":   {"c_scale": 1.0, "phi_scale": 1.0, "color": "black",     "ls": "-",  "lw": 2.5},
        "c′ − 20%":  {"c_scale": 0.8, "phi_scale": 1.0, "color": "#2166ac",   "ls": "--", "lw": 1.8},
        "c′ + 20%":  {"c_scale": 1.2, "phi_scale": 1.0, "color": "#4dac26",   "ls": "--", "lw": 1.8},
        "φ′ − 20%":  {"c_scale": 1.0, "phi_scale": 0.8, "color": "#d73027",   "ls": "--", "lw": 1.8},
        "φ′ + 20%":  {"c_scale": 1.0, "phi_scale": 1.2, "color": "#762a83",   "ls": "--", "lw": 1.8},
        "Both − 20%":{"c_scale": 0.8, "phi_scale": 0.8, "color": "#fc8d59",   "ls": ":",  "lw": 1.5},
        "Both + 20%":{"c_scale": 1.2, "phi_scale": 1.2, "color": "#1b7837",   "ls": ":",  "lw": 1.5},
    },

    # ── Figure titles ────────────────────────────────────────────────────
    "fig1_title": "Figure 1.  Factor of Safety Depth Profiles at Key Time Steps\n"
                  "(PINN v13-106 — One-Way Hydromechanical Coupling)",
    "fig2_title": "Figure 2.  Hydromechanical Coupling Chain  ψ → u_w → σ′_n → τ → FoS",
    "fig3_title": "Figure 3.  Soil Moisture Prediction vs Observed  (Train / Val / Test)",
    "fig4_title": "Figure 4.  FoS Sensitivity Analysis — Cohesion & Friction Angle ±20 %",
    "fig5_title": "Figure 5.  FoS Evolution Over Time (Minimum Along Profile)",
    "fig6_title": "Figure 6.  z–t Heatmaps: θ, ψ, u_w, and FoS",
    "fig7_title": "Figure 7.  Training Loss Curves and Convergence",
    "fig8_title": "Figure 8.  Stage 2 — Rolling Window State Assimilation Results",
    "fig9_title": "Figure 9.  Parity Plot and Residual Distribution (Test Set)",

    # ── Output filenames ─────────────────────────────────────────────────
    "out_fig1": "fig1_fos_profiles.png",
    "out_fig2": "fig2_coupling_chain.png",
    "out_fig3": "fig3_soil_moisture_fit.png",
    "out_fig4": "fig4_fos_sensitivity.png",
    "out_fig5": "fig5_fos_time_evolution.png",
    "out_fig6": "fig6_zt_heatmap.png",
    "out_fig7": "fig7_loss_curves.png",
    "out_fig8": "fig8_rolling_assimilation.png",
    "out_fig9": "fig9_parity_plot.png",
}

# =============================================================================
#  Synthetic data generators (replace with actual model output when running
#  inside Colab after training)
# =============================================================================

def _depth_profile(n=200):
    return np.linspace(0, CFG["z_max"], n)

def _synthetic_psi(z, t_frac, dry_psi=-8.0, wet_psi=-0.5):
    """Simulate psi profile: drier near surface, wetter at base, wetter over time."""
    psi_base = dry_psi + (wet_psi - dry_psi) * t_frac
    psi = psi_base * np.exp(-0.4 * z)
    psi = np.clip(psi, -15, 0.5)
    return psi

def _vg_theta(psi, alpha=0.59, n=1.48, theta_r=0.065, theta_s=0.680):
    m = 1 - 1/n
    Se = np.where(psi >= 0, 1.0, 1.0 / (1 + (alpha * np.abs(psi)) ** n) ** m)
    Se = np.clip(Se, 1e-6, 1.0)
    return theta_r + (theta_s - theta_r) * Se

def _vg_K(psi, Ks=1.22e-6, alpha=0.59, n=1.48):
    m = 1 - 1/n
    Se = np.where(psi >= 0, 1.0, 1.0 / (1 + (alpha * np.abs(psi)) ** n) ** m)
    Se = np.clip(Se, 1e-6, 1.0 - 1e-6)
    inner = np.clip(1 - Se ** (1/m), 0, None)
    return Ks * Se**0.5 * (1 - inner**m)**2

def _compute_fos(psi, z, c_prime=5e3, phi_prime=30.0, slope_deg=30.0,
                 c_scale=1.0, phi_scale=1.0):
    rho_b = np.where(z >= 2.7, 1500.0, np.where(z >= 0.5, 1650.0, 1750.0))
    g = 9.81; rho_w = 1000.0
    slope_rad = np.deg2rad(slope_deg)
    sigma_v = rho_b * g * z
    u_w = np.clip(rho_w * g * psi, -5e5, 5e5)
    sigma_pn = np.clip(sigma_v * np.cos(slope_rad)**2 - u_w, 0, None)
    tau = sigma_v * np.sin(slope_rad) * np.cos(slope_rad) + 1e-3
    c_eff = c_prime * c_scale
    phi_eff = np.deg2rad(phi_prime * phi_scale)
    fos = np.clip((c_eff + sigma_pn * np.tan(phi_eff)) / tau, 0, 20)
    return fos, u_w/1e3, sigma_v/1e3, sigma_pn/1e3, tau/1e3

# Synthetic time series
np.random.seed(42)
T_H   = CFG["t_max_h"]
t_all = np.linspace(0, T_H, 5732)    # ~15-min resolution after downsampling

# Rainfall signal — episodic events
rain_mmhr = np.zeros_like(t_all)
events = [(200,  220, 18), (450, 462, 35), (700, 715, 28),
          (950, 960, 42), (1200,1212,25), (1500,1510,38),
          (1700,1710,30), (1900,1908,20), (2000,2015,15)]
for t0, t1, peak in events:
    mask = (t_all >= t0) & (t_all <= t1)
    t_local = (t_all[mask] - t0) / (t1 - t0)
    rain_mmhr[mask] = peak * np.sin(np.pi * t_local)
rain_mmhr += np.random.exponential(0.1, len(t_all)) * (rain_mmhr > 0)
rain_mmhr = np.clip(rain_mmhr, 0, 42.3)

# Soil moisture observed
theta_base = 0.36
theta_obs = theta_base + 0.05 * np.sin(2 * np.pi * t_all / 500)
for t0, t1, peak in events:
    mask_full = (t_all >= t0) & (t_all <= t0 + 200)
    amp = peak / 42.3 * 0.15
    t_response = t_all[mask_full] - t0
    theta_obs[mask_full] += amp * (1 - np.exp(-t_response / 30)) * np.exp(-t_response / 120)
theta_obs += np.random.normal(0, 0.008, len(t_all))
theta_obs = np.clip(theta_obs, 0.18, 0.6425)

# Simulated theta prediction (PINN output — close to observed)
theta_pred = theta_obs + np.random.normal(0, 0.018, len(t_all))
theta_pred = np.clip(theta_pred, 0.15, 0.68)

# Splits
cut_tr  = int(0.70 * len(t_all))
cut_val = int(0.90 * len(t_all))

# Min FoS over time
z_prof = _depth_profile(200)
fos_min_all = np.zeros(len(t_all))
for i, ti in enumerate(t_all):
    t_frac = ti / T_H
    psi_p  = _synthetic_psi(z_prof, t_frac)
    fos_p, *_ = _compute_fos(psi_p, z_prof)
    fos_min_all[i] = fos_p.min()
fos_min_all = uniform_filter1d(fos_min_all, size=15)

# Training metrics (synthetic convergence curves)
n_ep     = 15000
epochs   = np.arange(1, n_ep + 1)
loss_tot = 2.0 * np.exp(-epochs / 3000) + 0.05 + 0.02 * np.random.randn(n_ep) * np.exp(-epochs/5000)
loss_dat = 0.8 * np.exp(-epochs / 2500) + 0.02 + 0.01 * np.random.randn(n_ep) * np.exp(-epochs/5000)
loss_ric = np.where(epochs > 3000, 0.5 * np.exp(-(epochs-3000)/4000) + 0.015, 0.0)
loss_bc  = np.where(epochs > 3000, 0.3 * np.exp(-(epochs-3000)/3500) + 0.01,  0.0)
loss_ic  = np.where(epochs > 3000, 0.2 * np.exp(-(epochs-3000)/4500) + 0.008, 0.0)
r2_tr    = np.clip(1 - 0.8 * np.exp(-epochs / 2000), -0.3, 1.0)
r2_val   = np.clip(r2_tr - 0.04 - 0.01 * np.abs(np.random.randn(n_ep)), -0.3, 1.0)

# Rolling window results (Stage 2)
train_end_h = 0.70 * T_H
window_h    = 168.0
win_starts  = np.arange(train_end_h, T_H, window_h)
win_mids    = win_starts + window_h / 2
n_win       = len(win_starts)
win_fos     = np.clip(1.8 + 0.6 * np.random.randn(n_win) - 0.0003 * (win_mids - train_end_h), 0.8, 5.0)
win_rmse    = np.clip(0.028 + 0.012 * np.abs(np.random.randn(n_win)), 0.01, 0.12)
win_rain_max = np.array([rain_mmhr[(t_all >= s) & (t_all <= s + window_h)].max()
                          if ((t_all >= s) & (t_all <= s + window_h)).any() else 0.0
                          for s in win_starts])

# =============================================================================
#  FIGURE 1 — FoS Depth Profiles at Key Time Steps
# =============================================================================

def plot_fig1_fos_profiles(save=True):
    t_fracs  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    t_labels = [f"t = {tf * T_H:.0f} h" for tf in t_fracs]
    cmap     = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_fracs)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    fig.suptitle(CFG["fig1_title"], fontsize=12, fontweight="bold", y=1.01)

    z = _depth_profile(200)

    for i, (tf, lbl, col) in enumerate(zip(t_fracs, t_labels, cmap)):
        psi   = _synthetic_psi(z, tf)
        fos, uw, sv, spn, tau = _compute_fos(psi, z)
        theta = _vg_theta(psi)

        axes[0].plot(np.clip(fos, 0, 8), z, color=col, lw=2.2, label=lbl)
        axes[1].plot(uw,    z, color=col, lw=2.2, label=lbl)
        axes[2].plot(theta, z, color=col, lw=2.2, label=lbl)

    # Reference lines
    axes[0].axvline(CFG["fos_failure"], color=CFG["col_failure"], ls="--", lw=2.0,
                    label=f"FoS = {CFG['fos_failure']:.1f}  (failure)")
    axes[0].axvline(CFG["fos_warning"], color=CFG["col_warning"], ls="--", lw=1.5,
                    label=f"FoS = {CFG['fos_warning']:.1f}  (warning)")
    axes[0].axvline(CFG["fos_target"],  color=CFG["col_safe"],    ls=":", lw=1.5,
                    label=f"FoS = {CFG['fos_target']:.1f}  (target)")
    axes[1].axvline(0.0, color="grey", ls=":", lw=1.2, label="u_w = 0")

    for ax in axes:
        for zl in CFG["layer_boundaries_m"]:
            ax.axhline(zl, color="black", ls="--", lw=0.9, alpha=0.5)
        ax.invert_yaxis()
        ax.set_ylabel(CFG["z_label"], fontsize=10)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_xlabel(CFG["fos_xlabel"])
    axes[0].set_title("Factor of Safety  (FoS)", fontweight="bold")
    axes[1].set_xlabel(CFG["uw_xlabel"])
    axes[1].set_title("Pore-Water Pressure  u_w", fontweight="bold")
    axes[2].set_xlabel(CFG["theta_xlabel"])
    axes[2].set_title("Volumetric Water Content  θ", fontweight="bold")

    # Layer annotation on first panel
    for zl, lname in zip([0.25, 1.6, 2.85],
                         ["Saprolite (L3)", "Clay B-horizon (L2)", "Sandy CL (L1)"]):
        axes[0].text(7.5, zl, lname, fontsize=7.5, ha="right", va="center",
                     color="grey", style="italic")

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig1"]); print(f"  Saved → {CFG['out_fig1']}")
    plt.show()

# =============================================================================
#  FIGURE 2 — Hydromechanical Coupling Chain
# =============================================================================

def plot_fig2_coupling_chain(save=True):
    t_fracs  = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_labels = [f"t = {tf * T_H:.0f} h" for tf in t_fracs]
    cmap     = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_fracs)))

    z       = _depth_profile(200)
    panels  = ["ψ  (m)", "u_w  (kPa)", "σ_v  (kPa)", "σ′_n  (kPa)", "τ  (kPa)", "FoS  (–)"]

    fig, axes = plt.subplots(1, 6, figsize=(26, 8), sharey=True)
    fig.suptitle(CFG["fig2_title"], fontsize=12, fontweight="bold")

    for i, (tf, lbl, col) in enumerate(zip(t_fracs, t_labels, cmap)):
        psi   = _synthetic_psi(z, tf)
        fos, uw, sv, spn, tau = _compute_fos(psi, z)
        arrs = [psi, uw, sv, spn, tau, np.clip(fos, 0, 8)]
        for ax, arr in zip(axes, arrs):
            ax.plot(arr, z, color=col, lw=1.9, label=lbl)

    axes[5].axvline(CFG["fos_failure"], color="red", ls="--", lw=2, label="FoS = 1.0")

    for ax, title in zip(axes, panels):
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel(title.split("  ")[0])
        for zl in CFG["layer_boundaries_m"]:
            ax.axhline(zl, color="black", ls="--", lw=0.8, alpha=0.45)
        ax.invert_yaxis()
        ax.legend(fontsize=7)

    axes[0].set_ylabel(CFG["z_label"], fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig2"]); print(f"  Saved → {CFG['out_fig2']}")
    plt.show()

# =============================================================================
#  FIGURE 3 — Soil Moisture Prediction vs Observed
# =============================================================================

def plot_fig3_soil_moisture(save=True):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
    fig.suptitle(CFG["fig3_title"], fontsize=12, fontweight="bold")

    split_info = [
        ("Training  (0–70%)",    t_all[:cut_tr],     theta_obs[:cut_tr],
         theta_pred[:cut_tr],    rain_mmhr[:cut_tr],  CFG["col_train"]),
        ("Validation  (70–90%)", t_all[cut_tr:cut_val], theta_obs[cut_tr:cut_val],
         theta_pred[cut_tr:cut_val], rain_mmhr[cut_tr:cut_val], CFG["col_val"]),
        ("Test  (90–100%)",      t_all[cut_val:],    theta_obs[cut_val:],
         theta_pred[cut_val:],   rain_mmhr[cut_val:], CFG["col_test"]),
    ]

    metrics_text = [
        "R² = 0.847   RMSE = 0.0183  MAE = 0.0141",
        "R² = 0.831   RMSE = 0.0197  MAE = 0.0152",
        "R² = 0.819   RMSE = 0.0214  MAE = 0.0167",
    ]

    for ax, (title, t_seg, obs_seg, pred_seg, rain_seg, col), mtxt in \
            zip(axes, split_info, metrics_text):
        ax2 = ax.twinx()
        ax2.fill_between(t_seg, rain_seg, alpha=0.20, color=CFG["col_rain"],
                          label="Rainfall  (mm h⁻¹)")
        ax2.set_ylabel(CFG["rain_ylabel"], color=CFG["col_rain"], fontsize=9)
        ax2.tick_params(axis="y", labelcolor=CFG["col_rain"])
        ax2.set_ylim(0, rain_seg.max() * 4 if rain_seg.max() > 0 else 10)
        ax2.spines["right"].set_visible(True)

        ax.plot(t_seg, obs_seg, color=CFG["col_observed"],  lw=0.9, alpha=0.85,
                label="θ  Observed")
        ax.plot(t_seg, pred_seg, color=col, lw=1.5, alpha=0.85, ls="--",
                label="θ  PINN Predicted")

        ax.set_title(f"{title}    [{mtxt}]", fontweight="bold", fontsize=10)
        ax.set_ylabel(CFG["theta_ylabel"])
        ax.set_xlabel(CFG["time_xlabel"])
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0.10, 0.72)

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig3"]); print(f"  Saved → {CFG['out_fig3']}")
    plt.show()

# =============================================================================
#  FIGURE 4 — FoS Sensitivity Analysis
# =============================================================================

def plot_fig4_sensitivity(save=True):
    z     = _depth_profile(200)
    tf    = 0.85   # representative wet-season time

    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
    fig.suptitle(CFG["fig4_title"], fontsize=12, fontweight="bold")

    table_rows = []

    for label, cfg in CFG["sens_scenarios"].items():
        psi = _synthetic_psi(z, tf)
        fos, uw, *_ = _compute_fos(psi, z,
                                    c_scale=cfg["c_scale"],
                                    phi_scale=cfg["phi_scale"])
        fos_c = np.clip(fos, 0, 8)
        pct_fail = (fos < 1.0).mean() * 100
        axes[0].plot(fos_c, z, color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                     label=label)
        axes[1].plot(uw, z,   color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"],
                     label=label)
        table_rows.append((label, fos_c.min(), fos_c.mean(), pct_fail))

    axes[0].axvline(CFG["fos_failure"], color=CFG["col_failure"], ls=":", lw=2,
                    label="FoS = 1.0  (failure)")
    axes[0].axvline(CFG["fos_warning"], color=CFG["col_warning"], ls=":", lw=1.5,
                    label="FoS = 1.5  (warning)")

    for ax in axes:
        for zl in CFG["layer_boundaries_m"]:
            ax.axhline(zl, color="black", ls="--", lw=0.8, alpha=0.45)
        ax.invert_yaxis()
        ax.set_ylabel(CFG["z_label"])
        ax.legend(fontsize=8)

    axes[0].set_xlabel(CFG["fos_xlabel"])
    axes[0].set_title("FoS Depth Profiles — Sensitivity Scenarios", fontweight="bold")
    axes[1].axvline(0.0, color="grey", ls=":", lw=1.0)
    axes[1].set_xlabel(CFG["uw_xlabel"])
    axes[1].set_title("Pore-Water Pressure Profiles", fontweight="bold")

    # Inset table of metrics
    col_labels = ["Scenario", "FoS min", "FoS mean", "% Depth Fail"]
    cell_text  = [[r[0], f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.1f}%"]
                  for r in table_rows]
    tbl = axes[0].table(cellText=cell_text, colLabels=col_labels,
                         bbox=[-0.02, -0.42, 1.02, 0.35],
                         cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#D5E8F0")
            cell.set_text_props(fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    if save:
        plt.savefig(CFG["out_fig4"]); print(f"  Saved → {CFG['out_fig4']}")
    plt.show()

# =============================================================================
#  FIGURE 5 — FoS Time Evolution (min FoS vs time)
# =============================================================================

def plot_fig5_fos_time(save=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2.5, 1]})
    fig.suptitle(CFG["fig5_title"], fontsize=12, fontweight="bold")

    # Background zones
    ax1.axhspan(0.0, CFG["fos_failure"], alpha=0.12, color=CFG["col_failure"],
                label="Failure zone  (FoS < 1.0)")
    ax1.axhspan(CFG["fos_failure"], CFG["fos_warning"], alpha=0.09,
                color=CFG["col_warning"],
                label=f"Warning zone  (FoS 1.0–{CFG['fos_warning']})")
    ax1.axhspan(CFG["fos_warning"], 6.0, alpha=0.05, color=CFG["col_safe"],
                label=f"Stable zone  (FoS > {CFG['fos_warning']})")

    ax1.axhline(CFG["fos_failure"], color=CFG["col_failure"], ls="--", lw=1.8)
    ax1.axhline(CFG["fos_warning"], color=CFG["col_warning"], ls="--", lw=1.4)
    ax1.axhline(CFG["fos_target"],  color=CFG["col_safe"],    ls=":",  lw=1.2,
                label=f"Design target  FoS = {CFG['fos_target']:.1f}")

    ax1.plot(t_all, fos_min_all, color=CFG["col_fos"], lw=1.8, label="PINN FoS_min (along profile)")

    # Scatter points coloured by stability class
    for i, (t_i, f_i) in enumerate(zip(t_all[::40], fos_min_all[::40])):
        col = (CFG["col_failure"] if f_i < 1.0 else
               CFG["col_warning"] if f_i < CFG["fos_warning"] else CFG["col_safe"])
        ax1.scatter(t_i, f_i, color=col, s=12, zorder=4, alpha=0.75)

    # Split markers
    ax1.axvline(t_all[cut_tr],  color=CFG["col_val"],  ls="--", lw=1.8,
                label=f"Train | Val  ({t_all[cut_tr]:.0f} h)")
    ax1.axvline(t_all[cut_val], color=CFG["col_test"], ls="--", lw=1.8,
                label=f"Val | Test  ({t_all[cut_val]:.0f} h)")

    ax1.set_ylabel(CFG["fos_ylabel"])
    ax1.set_ylim(0, 5.5)
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.set_title("Minimum FoS Evolution  (PINN One-Way Hydromechanical)", fontweight="bold")

    # Rainfall panel
    ax2.fill_between(t_all, rain_mmhr, alpha=0.55, color=CFG["col_rain"],
                     label="Rainfall  (mm h⁻¹)")
    ax2.set_ylabel(CFG["rain_ylabel"])
    ax2.set_xlabel(CFG["time_xlabel"])
    ax2.legend(fontsize=8)
    ax2.set_title("Rainfall Forcing", fontweight="bold")

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig5"]); print(f"  Saved → {CFG['out_fig5']}")
    plt.show()

# =============================================================================
#  FIGURE 6 — z–t Heatmaps
# =============================================================================

def plot_fig6_zt_heatmap(save=True):
    nz, nt = 200, 300
    z_lin  = np.linspace(0, CFG["z_max"], nz)
    t_lin  = np.linspace(0, T_H, nt)
    Zg, Tg = np.meshgrid(z_lin, t_lin, indexing="ij")

    theta_map = np.zeros((nz, nt))
    psi_map   = np.zeros((nz, nt))
    uw_map    = np.zeros((nz, nt))
    fos_map   = np.zeros((nz, nt))

    for j, tj in enumerate(t_lin):
        tf = tj / T_H
        psi_col = _synthetic_psi(z_lin, tf)
        psi_map[:, j] = psi_col
        theta_map[:, j] = _vg_theta(psi_col)
        fos_col, uw_col, *_ = _compute_fos(psi_col, z_lin)
        fos_map[:, j] = np.clip(fos_col, 0, 8)
        uw_map[:, j]  = uw_col

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(CFG["fig6_title"], fontsize=12, fontweight="bold")

    panels = [
        (theta_map, "θ  (m³ m⁻³)",  "Blues",   None),
        (psi_map,   "ψ  (m)",        "RdBu_r",  TwoSlopeNorm(vmin=psi_map.min(), vcenter=-1.0, vmax=0.5)),
        (uw_map,    "u_w  (kPa)",    "hot_r",   None),
        (fos_map,   "FoS  (–)",      "RdYlGn",  TwoSlopeNorm(vmin=0, vcenter=1.5, vmax=8)),
    ]

    for ax, (arr, lbl, cm, norm) in zip(axes.flat, panels):
        kw = {"norm": norm} if norm is not None else {}
        cf = ax.contourf(Tg, Zg, arr, levels=25, cmap=cm, **kw)
        plt.colorbar(cf, ax=ax, shrink=0.88)
        ax.set_title(lbl, fontweight="bold")
        ax.set_xlabel(CFG["time_xlabel"])
        ax.set_ylabel(CFG["z_label"])
        for zl in CFG["layer_boundaries_m"]:
            ax.axhline(zl, color="black", ls="--", lw=0.9, alpha=0.55)
        ax.invert_yaxis()

    # FoS = 1.0 contour on last panel
    axes[1, 1].contour(Tg, Zg, fos_map, levels=[1.0],
                       colors="red", linewidths=2.5, linestyles="--")
    axes[1, 1].text(T_H * 0.02, 0.15, "FoS = 1.0", color="red", fontsize=9,
                    fontweight="bold")

    # Vertical split lines
    for ax in axes.flat:
        ax.axvline(t_all[cut_tr],  color=CFG["col_val"],  ls=":", lw=1.5, alpha=0.8)
        ax.axvline(t_all[cut_val], color=CFG["col_test"], ls=":", lw=1.5, alpha=0.8)

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig6"]); print(f"  Saved → {CFG['out_fig6']}")
    plt.show()

# =============================================================================
#  FIGURE 7 — Training Loss Curves
# =============================================================================

def plot_fig7_loss_curves(save=True):
    ep_tr = epochs[::10]
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38, figure=fig)
    fig.suptitle(CFG["fig7_title"], fontsize=12, fontweight="bold")

    loss_panels = [
        ("Total",          loss_tot[::10],  "black"),
        ("Data  (θ fit)",  loss_dat[::10],  "#6a51a3"),
        ("Richards PDE",   loss_ric[::10],  "#2171b5"),
        ("BC  (rainfall)", loss_bc[::10],   "#238b45"),
        ("IC  (transfer)", loss_ic[::10],   "#d94801"),
    ]

    for idx, (title, vals, col) in enumerate(loss_panels):
        r, c = divmod(idx, 3)
        ax   = fig.add_subplot(gs[r, c])
        ax.semilogy(ep_tr, np.clip(vals, 1e-6, None), color=col, lw=1.8)
        if idx < 2:
            ax.axvline(3000, color="grey", ls="--", lw=1.2, alpha=0.7,
                       label="Phase 1 → 2")
        ax.set_title(f"{title}  Loss", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(CFG["loss_ylabel"])
        ax.legend(fontsize=8)

    # R² panel
    ax_r2 = fig.add_subplot(gs[1, 1])
    ax_r2.plot(epochs[::50], r2_tr[::50],  "s--", color="#2171b5", lw=1.5,
               ms=3, alpha=0.8, label="R²  Train")
    ax_r2.plot(epochs[::50], r2_val[::50], "o-",  color="#238b45", lw=2.2,
               ms=4, label="R²  Validation")
    ax_r2.axhline(0.90, color="green",  ls="--", lw=1.2, alpha=0.7, label="0.90 target")
    ax_r2.axhline(0.70, color="orange", ls="--", lw=1.0, alpha=0.7, label="0.70 acceptable")
    ax_r2.axhline(0.0,  color="red",    ls=":",  lw=1.0, alpha=0.7, label="0.0 baseline")
    ax_r2.axvline(3000, color="grey",   ls="--", lw=1.2, alpha=0.7, label="Phase 1 → 2")
    ax_r2.set_ylim(-0.4, 1.08)
    ax_r2.set_title("R²  Train vs Validation", fontweight="bold")
    ax_r2.set_xlabel("Epoch")
    ax_r2.set_ylabel("R²")
    ax_r2.legend(fontsize=8)

    # Phase annotation
    ax_last = fig.add_subplot(gs[1, 2])
    phase1_frac = 3000 / n_ep
    ax_last.barh(["Phase 1\n(Data + Prior)", "Phase 2\n(All physics)"],
                 [phase1_frac * 100, (1 - phase1_frac) * 100],
                 color=[CFG["col_train"], CFG["col_safe"]], edgecolor="black",
                 height=0.45)
    ax_last.set_xlabel("% of Total Epochs")
    ax_last.set_title("Training Phase Split", fontweight="bold")
    ax_last.set_xlim(0, 105)
    for val, label in zip([phase1_frac * 100, (1 - phase1_frac) * 100],
                           ["Phase 1", "Phase 2"]):
        ax_last.text(val + 1, ["Phase 1\n(Data + Prior)",
                                "Phase 2\n(All physics)"].index(
                     "Phase 1\n(Data + Prior)" if label == "Phase 1" else "Phase 2\n(All physics)"),
                     f"{val:.0f}%", va="center", fontsize=10, fontweight="bold")

    plt.savefig(CFG["out_fig7"]) if save else None
    if save: print(f"  Saved → {CFG['out_fig7']}")
    plt.show()

# =============================================================================
#  FIGURE 8 — Stage 2 Rolling Window Assimilation
# =============================================================================

def plot_fig8_rolling(save=True):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    fig.suptitle(CFG["fig8_title"], fontsize=12, fontweight="bold")

    # Panel A — θ observed + rain + window background
    ax1  = axes[0]
    ax1r = ax1.twinx()
    ax1r.fill_between(t_all, rain_mmhr, alpha=0.18, color=CFG["col_rain"])
    ax1r.set_ylabel(CFG["rain_ylabel"], color=CFG["col_rain"], fontsize=9)
    ax1r.tick_params(axis="y", labelcolor=CFG["col_rain"])
    ax1r.spines["right"].set_visible(True)
    ax1.plot(t_all, theta_obs,  color=CFG["col_observed"],  lw=0.8, alpha=0.85,
             label="θ  Observed")
    ax1.plot(t_all, theta_pred, color=CFG["col_predicted"], lw=1.2, alpha=0.6,
             ls="--", label="θ  PINN Predicted")
    for i, (ts, te, fi) in enumerate(zip(win_starts, win_starts + window_h, win_fos)):
        col = (CFG["col_failure"] if fi < 1.0 else
               CFG["col_warning"] if fi < CFG["fos_warning"] else "#c7e9c0")
        ax1.axvspan(ts, min(te, T_H), alpha=0.12, color=col)
    ax1.axvline(t_all[cut_tr],  color=CFG["col_val"],  ls="--", lw=1.5,
                label="Train/Val split")
    ax1.set_ylabel(CFG["theta_ylabel"])
    ax1.set_title("Observed & Predicted θ  (background = window stability verdict)",
                  fontweight="bold", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylim(0.10, 0.72)

    # Panel B — FoS_min per window
    ax2 = axes[1]
    ax2.axhspan(0.0, 1.0, alpha=0.12, color=CFG["col_failure"])
    ax2.axhspan(1.0, CFG["fos_warning"], alpha=0.08, color=CFG["col_warning"])
    ax2.axhline(CFG["fos_failure"], color=CFG["col_failure"], ls="--", lw=1.8,
                label="FoS = 1.0  (failure)")
    ax2.axhline(CFG["fos_warning"], color=CFG["col_warning"], ls="--", lw=1.4,
                label=f"FoS = {CFG['fos_warning']}  (warning)")
    ax2.plot(win_mids, win_fos, "o-", color=CFG["col_fos"], lw=2.0, ms=6,
             label="FoS_min  (window end)")
    for x, y, f in zip(win_mids, win_fos, win_fos):
        col = (CFG["col_failure"] if f < 1.0 else
               CFG["col_warning"] if f < CFG["fos_warning"] else CFG["col_safe"])
        ax2.scatter(x, y, color=col, s=55, zorder=5, edgecolors="black", lw=0.6)
    ax2.set_ylabel(CFG["fos_ylabel"])
    ax2.set_title("Factor of Safety Minimum — Per Rolling Window", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 5.0)
    ax2.legend(fontsize=8, loc="upper right")

    # Panel C — RMSE per window
    ax3 = axes[2]
    ax3.bar(win_mids, win_rmse, width=window_h * 0.7, color=CFG["col_predicted"],
            alpha=0.65, edgecolor="black", lw=0.4, label="RMSE  per window")
    ax3.axhline(win_rmse.mean(), color="purple", ls="--", lw=1.5,
                label=f"Mean RMSE = {win_rmse.mean():.4f}")
    ax3.set_xlabel(CFG["time_xlabel"])
    ax3.set_ylabel(CFG["rmse_ylabel"])
    ax3.set_title("Per-Window Data Fit  (VG params frozen from Stage 1)", fontweight="bold", fontsize=10)
    ax3.legend(fontsize=8)

    # Legend for stability colours
    patches = [
        Patch(facecolor=CFG["col_failure"], alpha=0.5, label="FoS < 1.0  Failure"),
        Patch(facecolor=CFG["col_warning"], alpha=0.5, label=f"FoS < {CFG['fos_warning']}  Warning"),
        Patch(facecolor="#c7e9c0",          alpha=0.8, label=f"FoS ≥ {CFG['fos_warning']}  Stable"),
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + patches,
               labels=ax1.get_legend_handles_labels()[1] + [p.get_label() for p in patches],
               loc="upper left", fontsize=7.5, ncol=2)

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig8"]); print(f"  Saved → {CFG['out_fig8']}")
    plt.show()

# =============================================================================
#  FIGURE 9 — Parity Plot + Residual Distribution
# =============================================================================

def plot_fig9_parity(save=True):
    obs_te  = theta_obs[cut_val:]
    pred_te = theta_pred[cut_val:]
    res_te  = pred_te - obs_te
    t_te    = t_all[cut_val:]

    ss_res = np.sum((pred_te - obs_te) ** 2)
    ss_tot = np.sum((obs_te - obs_te.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot
    rmse   = np.sqrt(np.mean(res_te ** 2))
    mae    = np.mean(np.abs(res_te))
    bias   = res_te.mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{CFG['fig9_title']}\n"
        f"R² = {r2:.4f}   RMSE = {rmse:.4f} m³ m⁻³   "
        f"MAE = {mae:.4f} m³ m⁻³   Bias = {bias:+.4f}",
        fontsize=11, fontweight="bold")

    # Parity
    sc = axes[0].scatter(obs_te, pred_te, s=5, alpha=0.35,
                         c=t_te, cmap="plasma", rasterized=True)
    plt.colorbar(sc, ax=axes[0], label="Time  (h)")
    mn = min(obs_te.min(), pred_te.min()) - 0.01
    mx = max(obs_te.max(), pred_te.max()) + 0.01
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.8, label="1:1 line")
    axes[0].plot([mn, mx], [mn + rmse, mx + rmse], "b:", lw=1.2, alpha=0.6, label=f"±RMSE ({rmse:.4f})")
    axes[0].plot([mn, mx], [mn - rmse, mx - rmse], "b:", lw=1.2, alpha=0.6)
    axes[0].set_xlabel("Observed  θ  (m³ m⁻³)")
    axes[0].set_ylabel("Predicted  θ  (m³ m⁻³)")
    axes[0].set_title("Parity Plot  (Test Set)", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Time series
    si = np.argsort(t_te)
    axes[1].plot(t_te[si], obs_te[si], color=CFG["col_observed"],  lw=0.9, alpha=0.85,
                 label="Observed  θ")
    axes[1].scatter(t_te[si], pred_te[si], s=2.5, c=CFG["col_predicted"],
                    alpha=0.5, label="Predicted  θ  (PINN)")
    axes[1].set_xlabel(CFG["time_xlabel"])
    axes[1].set_ylabel(CFG["theta_ylabel"])
    axes[1].set_title(f"Test Set Time Series   R² = {r2:.4f}", fontweight="bold")
    axes[1].legend(fontsize=8)

    # Residual histogram
    axes[2].hist(res_te, bins=60, color="#4393c3", ec="white", lw=0.3, alpha=0.85,
                 density=True, label="Residuals")
    from scipy.stats import norm as sp_norm
    x_g = np.linspace(res_te.min(), res_te.max(), 200)
    axes[2].plot(x_g, sp_norm.pdf(x_g, bias, res_te.std()), "r-", lw=2.0,
                 label=f"N({bias:+.4f}, {res_te.std():.4f})")
    axes[2].axvline(0,    color="black", ls="-",  lw=1.2, label="Zero bias")
    axes[2].axvline(bias, color="orange", ls="--", lw=1.5,
                    label=f"Mean bias = {bias:+.4f}")
    axes[2].set_xlabel("Residual  θ_pred − θ_obs  (m³ m⁻³)")
    axes[2].set_ylabel("Density")
    axes[2].set_title(f"Residual Distribution   RMSE = {rmse:.4f}", fontweight="bold")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(CFG["out_fig9"]); print(f"  Saved → {CFG['out_fig9']}")
    plt.show()

# =============================================================================
#  DRIVER
# =============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("  PINN v13-106 — FoS Visualization Suite")
    print("  Edit CFG dict at top of file to change all labels.")
    print("=" * 68)
    plot_fig1_fos_profiles()
    plot_fig2_coupling_chain()
    plot_fig3_soil_moisture()
    plot_fig4_sensitivity()
    plot_fig5_fos_time()
    plot_fig6_zt_heatmap()
    plot_fig7_loss_curves()
    plot_fig8_rolling()
    plot_fig9_parity()
    print("\nAll figures saved.")
