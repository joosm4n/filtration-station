"""Observe random objects for a short window, then check forecast accuracy at a later time.

Usage:
    python forecast_test.py
    python forecast_test.py --accel-max 0          # objects drift at constant velocity
    python forecast_test.py --dt 0.5 --qs 1e-8 1e-6 1e-4 1 --p0 100
    python forecast_test.py --plot                 # also save table images + charts

Methods compared on identical data:
  * Kalman filter, one run per process-noise value in --qs
  * Alpha-beta-gamma filter (TrackerConfig gains, or --abg A B G)
  * Least-squares line fit     (constant velocity)      <- baselines without a filter
  * Least-squares parabola fit (constant acceleration)

Results are printed as tables with full names and units. With --plot each table is also
saved as a PNG (plus summary charts) in --out-dir.

NOTE: the C++ API needs object ids from the caller (no data association yet),
so the test supplies the true ids.
"""

import argparse
import os
import sys
import textwrap

ap = argparse.ArgumentParser()
ap.add_argument(
    "--build-dir",
    default=os.environ.get(
        "FILTER_BUILD_DIR", "/home/Kieran/UNI/SpaceTeamsPRO/stp-cpp/build"
    ),
)
ap.add_argument("-n", "--objects", type=int, default=500)
ap.add_argument("--observe", type=float, default=5.0, help="observation window [s]")
ap.add_argument("--predict-at", type=float, default=120.0, help="forecast time [s]")
ap.add_argument("--dt", type=float, default=0.1, help="time between scans [s]")
ap.add_argument("--pos-range", type=float, default=5000.0, help="start position +-[m]")
ap.add_argument("--vel-max", type=float, default=2.0, help="speed per axis +-[m/s]")
ap.add_argument(
    "--accel-max", type=float, default=0.005, help="accel per axis +-[m/s^2]"
)
ap.add_argument("--qs", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-2, 1.0])
ap.add_argument("--p0", type=float, default=1.0, help="Kalman initial variance")
ap.add_argument("--abg", type=float, nargs=3, default=None, metavar=("A", "B", "G"))
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--plot", action="store_true", help="save table images and charts")
ap.add_argument("--out-dir", default="forecast_output")
ap.add_argument(
    "--accel-vec",
    type=float,
    nargs=3,
    default=[0.003, 0.0, 0.0],
    metavar=("AX", "AY", "AZ"),
    help="same acceleration for every object [m/s^2]",
)
args = ap.parse_args()

sys.path.insert(0, args.build_dir)
import numpy as np
import filter_bindings as fb

rng = np.random.default_rng(args.seed)
N = args.objects
SIG_POS, SIG_RC, SIG_VR, SIG_D = 2.0, 2.0, 0.02, 0.05  # sensor spec, 1-sigma
D_MIN, D_MAX = 5.0, 5000.0


# ===== table helpers (print + render to image) ====================================
class Table:
    def __init__(self, key, title, headers, rows, notes=(), best_row=None):
        self.key, self.title, self.headers = key, title, headers
        self.rows = [[str(c) for c in r] for r in rows]
        self.notes, self.best_row = list(notes), best_row


def _wrap(text, width):
    return textwrap.wrap(text, width) or [""]


def print_table(t, head_w=18, first_w=34):
    heads = [_wrap(h, first_w if i == 0 else head_w) for i, h in enumerate(t.headers)]
    nl = max(len(h) for h in heads)
    widths = [
        max(max(len(l) for l in heads[i]), max(len(r[i]) for r in t.rows))
        for i in range(len(heads))
    ]

    def line(cells):
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    rule = "-" * len(line([""] * len(widths)))
    print(f"\n{t.title}")
    print(rule)
    for k in range(nl):
        print(line([h[k] if k < len(h) else "" for h in heads]))
    print(rule)
    for i, r in enumerate(t.rows):
        print(line(r) + ("   <-- best" if i == t.best_row else ""))
    print(rule)
    for n in t.notes:
        print(
            textwrap.fill(
                n,
                width=max(len(rule), 90),
                initial_indent="  ",
                subsequent_indent="    ",
            )
        )


def render_table(t, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = [_wrap(h, 28 if i == 0 else 16) for i, h in enumerate(t.headers)]
    nl = max(len(h) for h in heads)
    widths = [
        max(max(len(l) for l in heads[i]), max(len(r[i]) for r in t.rows)) + 3
        for i in range(len(heads))
    ]
    note_lines = [l for n in t.notes for l in textwrap.wrap(n, 150)] or [""]
    row_in, head_in = 0.32, 0.2 * nl + 0.16
    table_in = head_in + row_in * len(t.rows)
    title_in = 0.5
    note_in = 0.21 * len(note_lines) + 0.15
    W = min(max(sum(widths) * 0.085, 6.0), 24.0)
    H = title_in + table_in + note_in

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.01, note_in / H, 0.98, table_in / H])
    ax.axis("off")
    tb = ax.table(
        cellText=t.rows,
        colLabels=["\n".join(h) for h in heads],
        colWidths=[w / sum(widths) for w in widths],
        cellLoc="center",
        loc="center",
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_height((head_in if r == 0 else row_in) / table_in)
        cell.set_edgecolor("#c8ccd4")
        if r == 0:
            cell.set_facecolor("#2b3a55")
            cell.set_text_props(color="white", weight="bold")
        elif r - 1 == t.best_row:
            cell.set_facecolor("#dff3e0")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f6f7f9" if r % 2 else "white")
    fig.text(0.012, 1 - 0.32 / H, t.title, fontsize=12, weight="bold", va="center")
    fig.text(
        0.012, 0.08 / H, "\n".join(note_lines), fontsize=8, va="bottom", color="#444"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def f1(x):
    return f"{x:,.1f}"


# ===== random truth ================================================================
x0 = rng.uniform(-args.pos_range, args.pos_range, (N, 3))
v0 = rng.uniform(-args.vel_max, args.vel_max, (N, 3))
a0 = rng.uniform(-args.accel_max, args.accel_max, (N, 3))
diam = rng.uniform(D_MIN, D_MAX, N)


def truth(t):
    return x0 + v0 * t + 0.5 * a0 * t * t


n_scans = int(round(args.observe / args.dt)) + 1
times = np.arange(n_scans) * args.dt  # 0, dt, ..., observe
pos = np.stack([truth(t) for t in times])  # (T, N, 3)
vel = np.stack([v0 + a0 * t for t in times])
rho = np.linalg.norm(pos, axis=2)
rc = rho - 0.5 * diam
vr = np.einsum("tni,tni->tn", pos, vel) / rho

meas = pos + rng.normal(0, SIG_POS, pos.shape)  # noisy positions (T, N, 3)

rows = np.empty((n_scans, N, 8))
rows[..., 0] = np.arange(N)
rows[..., 1] = times[:, None]
rows[..., 2:5] = meas
rows[..., 5] = rc + rng.normal(0, SIG_RC, rc.shape)
rows[..., 6] = vr + rng.normal(0, SIG_VR, vr.shape)
rows[..., 7] = diam * (1 + rng.normal(0, SIG_D, (n_scans, N)))
flat = rows.reshape(-1, 8)

T_PRED = args.predict_at
true_future = truth(T_PRED)


# ===== methods =====================================================================
def run_bank(cfg):
    bank = fb.TrackerBank(cfg, threads=args.threads)
    rejected = bank.ingest_array(flat)
    assert rejected == 0, f"{rejected} measurements rejected"
    ps = bank.predict_all(T_PRED)
    assert [p.id for p in ps] == list(range(N))
    xyz = np.array([[p.x, p.y, p.z] for p in ps])
    sig = np.array([[p.sx, p.sy, p.sz] for p in ps])
    dia = np.array([bank.diameter(i) for i in range(N)])  # same for every filter
    return xyz, sig, dia


def poly_fit(deg):
    A = np.vander(times, deg + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(A, meas.reshape(n_scans, -1), rcond=None)
    return ((T_PRED ** np.arange(deg + 1)) @ coef).reshape(N, 3)


methods = {}  # short -> dict(full, xyz, sig)

for q in args.qs:
    cfg = fb.TrackerConfig()
    cfg.filter = fb.FilterType.KALMAN
    cfg.q, cfg.p0, cfg.v0 = q, args.p0, 0.0
    xyz, sig, dia_est = run_bank(cfg)
    methods[f"Kalman q={q:g}"] = dict(
        full=f"Kalman filter, process noise q = {q:g} m\u00b2/s\u2075",
        xyz=xyz,
        sig=sig,
    )

cfg = fb.TrackerConfig()
cfg.filter = fb.FilterType.ABG
if args.abg:
    cfg.alpha, cfg.beta, cfg.gamma = args.abg
xyz, sig, dia_est = run_bank(cfg)
methods["ABG"] = dict(
    full=f"Alpha-beta-gamma filter (\u03b1={cfg.alpha:g}, \u03b2={cfg.beta:g}, \u03b3={cfg.gamma:g})",
    xyz=xyz,
    sig=sig,
)
methods["LSQ line"] = dict(
    full="Least-squares line fit (constant velocity)", xyz=poly_fit(1), sig=None
)
methods["LSQ parabola"] = dict(
    full="Least-squares parabola fit (constant acceleration)", xyz=poly_fit(2), sig=None
)

# ===== metrics =====================================================================
errs, zstats = {}, {}
for k, m in methods.items():
    e = m["xyz"] - true_future
    errs[k] = np.linalg.norm(e, axis=1)
    s = m["sig"]
    if s is not None and np.all(np.isfinite(s)) and np.all(s > 0):
        z = e / s
        zstats[k] = (np.mean(z**2), np.mean(np.abs(z) < 3) * 100)
best = min(errs, key=lambda k: np.median(errs[k]))

rc_true = np.linalg.norm(true_future, axis=1) - 0.5 * diam
rc_err = {
    k: np.abs(np.linalg.norm(m["xyz"], axis=1) - 0.5 * dia_est - rc_true)
    for k, m in methods.items()
}
best_rc = min(rc_err, key=lambda k: np.median(rc_err[k]))

# ===== tables ======================================================================
tables = []

setup_rows = [
    ("Number of objects", N, "count"),
    ("Observation window", f"{args.observe:g}", "s"),
    ("Time between scans", f"{args.dt:g}", "s"),
    ("Scans per object", n_scans, "count"),
    ("Total measurements ingested", f"{len(flat):,}", "count"),
    ("Forecast time (since simulation start)", f"{T_PRED:g}", "s"),
    ("Look-ahead beyond last report", f"{T_PRED - times[-1]:g}", "s"),
    ("Initial position range, per axis", f"\u00b1{args.pos_range:g}", "m"),
    ("Initial speed range, per axis", f"\u00b1{args.vel_max:g}", "m/s"),
    ("Acceleration range, per axis", f"\u00b1{args.accel_max:g}", "m/s\u00b2"),
    ("True diameter range", f"{D_MIN:g} to {D_MAX:g}", "m"),
    ("Sensor noise, position x/y/z (1\u03c3)", f"{SIG_POS:g}", "m"),
    ("Sensor noise, closest-point distance r_c (1\u03c3)", f"{SIG_RC:g}", "m"),
    ("Sensor noise, range rate v_r (1\u03c3)", f"{SIG_VR:g}", "m/s"),
    ("Sensor noise, diameter d (1\u03c3)", f"{SIG_D * 100:g}", "% of true diameter"),
    ("Kalman initial state variance p0", f"{args.p0:g}", "per state component"),
    ("Worker threads", args.threads, "count"),
    ("Random seed", args.seed, "-"),
]
tables.append(
    Table("1_setup", "Test setup", ["Parameter", "Value", "Unit"], setup_rows)
)

pos_rows = []
for k, m in methods.items():
    e = errs[k]
    z = zstats.get(k)
    pos_rows.append(
        [
            m["full"],
            f1(e.mean()),
            f1(np.median(e)),
            f1(np.percentile(e, 95)),
            f1(e.max()),
            f"{z[0]:.2f}" if z else "n/a",
            f"{z[1]:.1f}" if z else "n/a",
        ]
    )
ref_median = np.median(np.linalg.norm(truth(times[-1]) - true_future, axis=1))
vel_ref = SIG_POS * np.sqrt(12 / n_scans) / args.observe * T_PRED
tables.append(
    Table(
        "2_position",
        f"Centre-of-mass position forecast error at t = {T_PRED:g} s",
        [
            "Method",
            "Mean 3D position error [m]",
            "Median 3D position error [m]",
            "95th percentile 3D position error [m]",
            "Maximum 3D position error [m]",
            "Mean normalised squared error z\u00b2 [ideal = 1]",
            "Errors within 3\u03c3 of own uncertainty [%] [ideal = 99.7]",
        ],
        pos_rows,
        notes=[
            "3D position error = Euclidean distance between the forecast and the true centre of mass.",
            "z = per-axis error divided by the filter's own predicted 1\u03c3 position uncertainty; "
            "n/a for methods that give no uncertainty. z\u00b2 well above 1 means overconfident.",
            f"Reference: assuming the object stays at its last reported position gives a median error of "
            f"{ref_median:,.0f} m. The velocity uncertainty of a straight-line fit alone is about "
            f"{vel_ref:,.0f} m per axis at this look-ahead.",
        ],
        best_row=list(methods).index(best),
    )
)

rel = (dia_est - diam) / diam
ab = np.abs(rel)
tables.append(
    Table(
        "3_diameter",
        "Diameter estimate accuracy (running mean of reports; identical for every filter)",
        ["Quantity", "Value", "Unit"],
        [
            ("Mean absolute relative error", f"{ab.mean() * 100:.2f}", "%"),
            (
                "95th percentile absolute relative error",
                f"{np.percentile(ab, 95) * 100:.2f}",
                "%",
            ),
            ("Maximum absolute relative error", f"{ab.max() * 100:.2f}", "%"),
            (
                "Single-report noise (sensor spec, 1\u03c3)",
                f"{SIG_D * 100:.2f}",
                "% of true diameter",
            ),
            (
                f"Expected 1\u03c3 error after {n_scans} reports",
                f"{SIG_D / np.sqrt(n_scans) * 100:.2f}",
                "% of true diameter",
            ),
            ("Mean absolute error", f1(np.abs(dia_est - diam).mean()), "m"),
            ("Maximum absolute error", f1(np.abs(dia_est - diam).max()), "m"),
        ],
    )
)

rc_rows = []
for k, m in methods.items():
    e = rc_err[k]
    rc_rows.append(
        [
            m["full"],
            f1(e.mean()),
            f1(np.median(e)),
            f1(np.percentile(e, 95)),
            f1(e.max()),
        ]
    )
n_inside = int(np.sum(rc_true < 0))
rc_notes = [
    "Closest-point distance r_c = distance from the shuttle (origin) to the object's surface = "
    "|position| \u2212 diameter/2. Forecast uses the forecast position and the estimated diameter.",
]
if n_inside:
    rc_notes.append(
        f"{n_inside} objects have true r_c < 0 at t = {T_PRED:g} s (the shuttle would be inside them)."
    )
tables.append(
    Table(
        "4_closest_point",
        f"Closest-point distance r_c forecast error at t = {T_PRED:g} s",
        [
            "Method",
            "Mean absolute r_c error [m]",
            "Median absolute r_c error [m]",
            "95th percentile absolute r_c error [m]",
            "Maximum absolute r_c error [m]",
        ],
        rc_rows,
        notes=rc_notes,
        best_row=list(methods).index(best_rc),
    )
)

edges = np.quantile(diam, [0, 1 / 3, 2 / 3, 1])
band_rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (diam >= lo) & (diam <= hi)
    band_rows.append(
        [
            f"{lo:,.0f} to {hi:,.0f}",
            int(msk.sum()),
            f1(np.median(errs[best][msk])),
            f1(np.median(rc_err[best][msk])),
            f1(np.median(0.5 * np.abs(dia_est[msk] - diam[msk]))),
        ]
    )
tables.append(
    Table(
        "5_by_size",
        f"Accuracy by object size, best position method: {methods[best]['full']}",
        [
            "True diameter band [m]",
            "Objects [count]",
            "Median 3D position error [m]",
            "Median absolute r_c error [m]",
            "Part of r_c error caused by diameter estimate [m]",
        ],
        band_rows,
        notes=[
            "Objects are split into three equally sized bands by true diameter. "
            "The last column is the median of 0.5 \u00d7 |estimated diameter \u2212 true diameter|."
        ],
    )
)

# ===== output ======================================================================
print(
    f"\n{N} objects observed for {args.observe:g} s, forecast at t = {T_PRED:g} s "
    f"(sensor noise and truth model as in table 1). Best row in each method table is marked."
)
for t in tables:
    print_table(t)

if args.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    for t in tables:
        render_table(t, os.path.join(args.out_dir, f"table_{t.key}.png"))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    hi = max(np.percentile(v, 99) for v in errs.values())
    bins = np.logspace(
        np.log10(max(min(v.min() for v in errs.values()), 1e-2)), np.log10(hi), 50
    )
    for k, e3 in errs.items():
        ax[0].hist(e3, bins=bins, histtype="step", label=methods[k]["full"])
    ax[0].set_xscale("log")
    ax[0].set_xlabel(f"3D position forecast error at t = {T_PRED:g} s [m]")
    ax[0].set_ylabel("Number of objects")
    ax[0].set_title("Distribution of forecast error")
    ax[0].legend(fontsize=6)

    names = list(methods)
    xs = np.arange(len(names))
    ax[1].bar(
        xs - 0.2,
        [np.median(errs[k]) for k in names],
        0.4,
        label="Median 3D position error",
    )
    ax[1].bar(
        xs + 0.2,
        [np.median(rc_err[k]) for k in names],
        0.4,
        label="Median absolute r_c error",
    )
    ax[1].set_yscale("log")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax[1].set_ylabel("Error [m]")
    ax[1].set_title(f"Median error by method at t = {T_PRED:g} s")
    ax[1].legend(fontsize=8)

    sc = ax[2].scatter(
        true_future[:, 0],
        true_future[:, 1],
        c=errs[best],
        s=8,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
    )
    fig.colorbar(sc, ax=ax[2], label="3D position error [m]")
    ax[2].set_title(f"{best}: error by true position")
    ax[2].set_xlabel(f"True x at t = {T_PRED:g} s [m]")
    ax[2].set_ylabel(f"True y at t = {T_PRED:g} s [m]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "charts.png"), dpi=130)
    plt.close(fig)
    print(
        f"\nSaved {len(tables)} table images and charts.png to {os.path.abspath(args.out_dir)}"
    )
