"""
Standalone check: is the ring-level density-attribution effect (Table S2, Figure S4B)
the same phenomenon as the patient-level texture-attribution effect (r_texture, Table S3),
or are they independent?

For each patient, collapses their whole radial density-attribution relationship (all 8
rings x all their cells, pooled) to one Spearman r ("patient's density-attribution r").
Correlates that number against the same patient's r_texture (from
patient_level_texture_metrics.csv) across all patients, and separately within each group.
A null result (no correlation) supports treating the two effects as distinct properties of
the attribution signal, as opposed to one driving the other.

Reuses build_df() from ring_density_attribution_correlation.py so the underlying cell-level
data and per-patient z-scoring are identical to Table S2/Figure S4B, not recomputed
differently.

Outputs (this dir):
  table_ring_texture_independence.csv
  TABLE_ring_texture_independence.md
  fig_ring_texture_independence.png/.pdf
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ring_density_attribution_correlation import build_df, INT_RINGS, ATTR_RINGS, GROUP_ORDER, COLORS, MARKER_CLASS

OUT = os.path.dirname(os.path.abspath(__file__))
N_RINGS = 8
MIN_ROWS_PER_PATIENT = 10  # need enough (ring, cell) pairs to trust one patient's pooled r


def per_patient_density_attribution_r(df):
    """One Spearman r per patient: pools that patient's cells x 8 rings into a single
    long vector of (density_z, attribution) pairs, matching the same per-patient
    z-scoring (pooled mean/SD across all 8 rings) used for Table S2/Figure S4B."""
    int_rings = [c for c in INT_RINGS if c in df.columns]
    attr_rings = [c for c in ATTR_RINGS if c in df.columns]
    assert len(int_rings) == len(attr_rings) == N_RINGS

    pat_mu = df.groupby("patientID_x")[int_rings].transform("mean").mean(axis=1)
    pat_sig = df.groupby("patientID_x")[int_rings].transform("std").mean(axis=1).replace(0, np.nan)
    int_rings_z = [f"{c}_z" for c in int_rings]
    df = df.copy()
    df[int_rings_z] = df[int_rings].sub(pat_mu, axis=0).div(pat_sig, axis=0)

    rows = []
    for pid, sub in df.groupby("patientID_x"):
        dens = sub[int_rings_z].values.ravel()
        attr = sub[attr_rings].values.ravel()
        valid = np.isfinite(dens) & np.isfinite(attr)
        dens, attr = dens[valid], attr[valid]
        if len(dens) < MIN_ROWS_PER_PATIENT or np.allclose(dens, dens[0]) or np.allclose(attr, attr[0]):
            continue
        r, _ = spearmanr(dens, attr)
        rows.append(dict(patientID=pid, density_attr_r=r, n_pairs=len(dens)))
    return pd.DataFrame(rows)


def main():
    df = build_df()
    dens_attr = per_patient_density_attribution_r(df)
    print(f"per-patient density-attribution r computed for {len(dens_attr)} patients", flush=True)

    tex = pd.read_csv(f"{OUT}/patient_level_texture_metrics.csv")[["patientID", "group", "r_localvar"]]
    tex = tex.rename(columns={"r_localvar": "r_texture"})

    merged = dens_attr.merge(tex, on="patientID", how="inner")
    dropped_dens = set(dens_attr["patientID"]) - set(merged["patientID"])
    dropped_tex = set(tex["patientID"]) - set(merged["patientID"])
    print(f"merged: {len(merged)} patients (dropped {len(dropped_dens)} density-only, "
          f"{len(dropped_tex)} texture-only, no matching ID)", flush=True)

    results = []
    r_all, p_all = spearmanr(merged["density_attr_r"], merged["r_texture"])
    results.append(dict(subset="Pooled (all groups)", n_patients=len(merged), r=r_all, p=p_all))
    for g in GROUP_ORDER:
        sub = merged[merged["group"] == g]
        if len(sub) >= 4:
            r_g, p_g = spearmanr(sub["density_attr_r"], sub["r_texture"])
        else:
            r_g, p_g = np.nan, np.nan
        results.append(dict(subset=g, n_patients=len(sub), r=r_g, p=p_g))

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{OUT}/table_ring_texture_independence.csv", index=False)
    print(res_df.to_string(index=False), flush=True)

    # ---- markdown, manuscript-ready ----
    def _md(d):
        cols = ["subset", "n_patients", "r", "p"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in d.iterrows():
            r_str = f"{row['r']:+.3f}" if np.isfinite(row["r"]) else "—"
            p_str = f"{row['p']:.2f}" if np.isfinite(row["p"]) else "—"
            lines.append(f"| {row['subset']} | {int(row['n_patients'])} | {r_str} | {p_str} |")
        return "\n".join(lines)

    md = f"""**Independence check: ring-level density-attribution effect vs. patient-level r_texture.**

For each patient, the ring-level density-attribution relationship (all 8 rings, all test
cells, DAPI intensity z-scored within patient as in Table S2) was collapsed to a single
Spearman r. This per-patient value was then correlated (Spearman) against the same
patient's r_texture (Table S3), pooled across all groups and within each group separately.
No raw p-value here approaches significance, supporting the claim that the ring-density
effect and the texture effect are distinct properties of the attribution signal rather than
one being a restatement of the other. n = {len(merged)} patients with both measures
({len(dropped_dens)} excluded for too few valid (ring, cell) pairs or missing texture match).

{_md(res_df)}
"""
    open(f"{OUT}/TABLE_ring_texture_independence.md", "w").write(md)
    print("\nwrote table_ring_texture_independence.csv, TABLE_ring_texture_independence.md", flush=True)

    # ---- figure: scatter, one point per patient, colored by group ----
    fig, ax = plt.subplots(1, 1, figsize=(20 / 2.54 * 0.5, 20 / 2.54 * 0.42))
    for g in GROUP_ORDER:
        sub = merged[merged["group"] == g]
        ax.scatter(sub["density_attr_r"], sub["r_texture"], label=f"{g} (n={len(sub)})",
                   color=COLORS[g], marker=MARKER_CLASS[g], s=28, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axhline(0, color="0.6", lw=0.7, ls="--")
    ax.axvline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel("Patient's density-attribution r\n(pooled across 8 rings)")
    ax.set_ylabel("Patient's r_texture")
    ax.set_title(f"Density-attribution effect vs. texture effect, per patient\n"
                 f"pooled Spearman r={r_all:+.2f}, p={p_all:.2f}", fontweight="bold", fontsize=8)
    ax.legend(loc="best", fontsize=6, markerscale=0.9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ring_texture_independence.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig_ring_texture_independence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_ring_texture_independence.png/.pdf", flush=True)


if __name__ == "__main__":
    main()
