from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(os.environ.get("PENTARNION_DATA_ROOT", str(ROOT/"data"))).resolve()
DATA = INPUT / "spin_dense"
FIG = Path(os.environ.get('PENTARNION_OUTPUT_DIR', str(ROOT/'outputs/plots'))).expanduser().resolve()
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def select_full_horizon(df: pd.DataFrame, max_points: int = 120000) -> pd.DataFrame:
    """Retain the complete post-transient time horizon in temporal order.

    Uniform decimation is used only when needed to keep the rasterised line
    figure manageable.  The first sample, final sample, and full time span are
    always retained; no interpolation or geometric smoothing is introduced.
    """
    q = df.copy()
    if len(q) > max_points:
        idx = np.linspace(0, len(q) - 1, max_points, dtype=int)
        q = q.iloc[np.unique(idx)].copy()
    return q.reset_index(drop=True)


def add_connected_trajectory(ax, df: pd.DataFrame, cols, linewidth: float, alpha: float):
    """Draw one continuous time-ordered curve over the full retained horizon.

    No scatter markers, point-cloud rendering, interpolation, or geometric smoothing
    are used.  Uniformly decimated samples, when required for rendering, preserve
    the complete time span and temporal order.
    """
    xyz = df.loc[:, cols].to_numpy(float)
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        linewidth=linewidth,
        alpha=alpha,
        solid_capstyle="round",
        solid_joinstyle="round",
        rasterized=True,
    )
    # Emphasise one contiguous terminal arc so the path remains visibly a
    # trajectory rather than being mistaken for a point cloud.  The pale line
    # still spans the complete retained horizon.
    n_highlight = min(5000, max(1200, len(xyz) // 20))
    h = xyz[-n_highlight:]
    ax.plot(
        h[:, 0], h[:, 1], h[:, 2],
        linewidth=max(0.34, 4.0 * linewidth),
        alpha=0.96,
        solid_capstyle="round",
        solid_joinstyle="round",
        rasterized=True,
        zorder=4,
    )
    for values, setter in zip(xyz.T, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        pad = 0.04 * max(hi - lo, 1e-9)
        setter(lo - pad, hi + pad)
    return float(df["t"].iloc[-1] - df["t"].iloc[0])


scan = pd.read_csv(INPUT / "spin" / "representative_lyapunov_scan.csv")
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.plot(scan["Omega"], scan["lambda_max"], marker="o")
ax.axhline(0, linewidth=0.9)
ax.set_xlabel(r"switching frequency $\Omega$")
ax.set_ylabel(r"largest numerical Lyapunov exponent $\lambda_{\max}$")
ax.set_title("Switching-spin frequency response")
ax.grid(True)
save(fig, "spin_lyapunov_frequency_scan.png")

for tag, title, linewidth, alpha in [
    ("regular", r"Regular regime, $\Omega=0.30$", 0.075, 0.20),
    ("chaotic", r"Positive-exponent regime, $\Omega=1.20$", 0.065, 0.13),
]:
    # The trajectory panel uses a long consecutive post-transient interval,
    # drawn as one time-ordered line.  The component panel keeps the previously
    # selected late-time interval so that individual oscillations remain readable.
    long_record = pd.read_csv(DATA / f"{tag}_timeseries.csv")
    trajectory = select_full_horizon(long_record, max_points=120000)
    display = pd.read_csv(DATA / f"{tag}_display_window.csv")
    poincare = pd.read_csv(DATA / f"{tag}_poincare.csv")
    tau_display = display["t"] - display["t"].iloc[0]

    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    add_connected_trajectory(
        ax, trajectory, ["y_h", "y_b", "y_c"], linewidth=linewidth, alpha=alpha
    )
    ax.set_xlabel(r"$y_h$")
    ax.set_ylabel(r"$y_b$")
    ax.set_zlabel(r"$y_c$")
    ax.set_title(title + " - full-horizon connected trajectory")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-56)
    save(fig, f"spin_{tag}_trajectory_dense.png")

    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    ax.scatter(
        poincare["y_h"],
        poincare["y_b"],
        s=3 if tag == "regular" else 1.6,
        alpha=0.65,
        rasterized=True,
    )
    ax.set_xlabel(r"$y_h$")
    ax.set_ylabel(r"$y_b$")
    ax.set_title(title + " - long stroboscopic section")
    ax.grid(True)
    save(fig, f"spin_{tag}_poincare_dense.png")

    fig, ax = plt.subplots(figsize=(10.3, 5.5))
    ax.plot(tau_display, display["y_h"], label=r"$y_h$", linewidth=0.8)
    ax.plot(tau_display, display["y_b"], label=r"$y_b$", linewidth=0.8)
    ax.plot(tau_display, display["y_c"], label=r"$y_c$", linewidth=0.8)
    ax.set_xlabel("elapsed time in displayed interval")
    ax.set_ylabel("component")
    ax.set_title(title + " - representative component record")
    ax.legend(ncol=3)
    ax.grid(True)
    save(fig, f"spin_{tag}_timeseries_dense.png")

# Fixed-frame projection of the same long positive-exponent interval.
chaotic_long = pd.read_csv(DATA / "chaotic_timeseries.csv")
chaotic_trajectory = select_full_horizon(chaotic_long, max_points=120000)
fig = plt.figure(figsize=(8.2, 6.8))
ax = fig.add_subplot(111, projection="3d")
add_connected_trajectory(ax, chaotic_trajectory, ["a", "b", "d"], linewidth=0.060, alpha=0.13)
ax.set_xlabel(r"$a$")
ax.set_ylabel(r"$b$")
ax.set_zlabel(r"$\ell$")
ax.set_title("Positive-exponent run - full-horizon connected fixed-frame trajectory")
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=24, azim=-56)
save(fig, "spin_chaotic_fixed_frame_dense.png")

# Exact switching-work audit from the long retained records.
for tag in ["regular", "chaotic"]:
    audit = pd.read_csv(DATA / f"{tag}_work_audit.csv")
    audit["tau"] = audit["t"] - audit["t"].iloc[0]
    q = audit[audit["tau"] >= audit["tau"].iloc[-1] - 300]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.plot(q["tau"], q["H_minus_H0"], label=r"$H(t)-H(t_0)$", linewidth=0.9)
    ax.plot(
        q["tau"],
        q["work_integral"],
        label=r"$\int_{t_0}^t \partial_\gamma H\,\dot\gamma\,ds$",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xlabel(r"$t-t_0$")
    ax.set_ylabel("energy change / accumulated work")
    ax.set_title(f"Exact switching-work identity audit ({tag} run)")
    ax.legend()
    ax.grid(True)
    save(fig, f"spin_energy_balance_{tag}.png")

(FIG / "spin_energy_balance_dense.png").write_bytes(
    (FIG / "spin_energy_balance_chaotic.png").read_bytes()
)
print("wrote full-horizon connected spin trajectories and representative time-series panels")
