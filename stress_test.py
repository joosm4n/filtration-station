"""Stress test for filter_bindings.TrackerBank with many objects.

Usage:
    python stress_test.py                       # 2000 objects, 60 scans
    python stress_test.py -n 20000 --steps 100 --threads 1 2 4 8

Checks:
  1. correctness : nothing rejected, prediction error vs. truth, sigma calibration
  2. determinism : identical predictions for every thread count
  3. mode        : scan-by-scan ingest vs. one big shuffled batch give the same result
  4. speed       : measurements/s for ingest and predict_all per thread count
  5. edge cases  : duplicate time, NaN position, past prediction, unknown id

NOTE: the C++ API needs object ids from the caller (no data association yet),
so the test supplies the true ids.
"""

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument(
    "--build-dir",
    default=os.environ.get(
        "FILTER_BUILD_DIR", "/home/Kieran/UNI/SpaceTeamsPRO/stp-cpp/build"
    ),
)
ap.add_argument("-n", "--objects", type=int, default=2000)
ap.add_argument("--steps", type=int, default=60, help="scans per object")
ap.add_argument("--dt", type=float, default=1.0)
ap.add_argument("--horizon", type=float, default=5.0, help="look-ahead [s]")
ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--filter", choices=["kalman", "abg"], default="kalman")
ap.add_argument("--q", type=float, default=None, help="Kalman process noise")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

sys.path.insert(0, args.build_dir)
import numpy as np
import filter_bindings as fb

rng = np.random.default_rng(args.seed)
N, T, DT = args.objects, args.steps, args.dt
SIG_POS, SIG_RC, SIG_VR, SIG_D = 2.0, 2.0, 0.02, 0.05

# ---- synthetic truth: constant-acceleration bodies around the shuttle ---------------
ids = np.arange(N, dtype=np.int64)
x0 = rng.uniform(-5000, 5000, (N, 3))
v0 = rng.uniform(-2.0, 2.0, (N, 3))
a0 = rng.uniform(-0.005, 0.005, (N, 3))
diam = rng.uniform(5.0, 5000.0, N)


def truth(t):
    """Positions (N,3) at scalar time t."""
    return x0 + v0 * t + 0.5 * a0 * t * t


times = np.arange(T) * DT
pos = np.stack([truth(t) for t in times])  # (T, N, 3)
vel = np.stack([v0 + a0 * t for t in times])
rho = np.linalg.norm(pos, axis=2)
rc = rho - 0.5 * diam
vr = np.einsum("tni,tni->tn", pos, vel) / rho

rows = np.empty((T, N, 8))
rows[..., 0] = ids
rows[..., 1] = times[:, None]
rows[..., 2:5] = pos + rng.normal(0, SIG_POS, pos.shape)
rows[..., 5] = rc + rng.normal(0, SIG_RC, rc.shape)
rows[..., 6] = vr + rng.normal(0, SIG_VR, vr.shape)
rows[..., 7] = diam * (1 + rng.normal(0, SIG_D, (T, N)))
total = T * N
print(f"{N} objects x {T} scans = {total:,} measurements, filter={args.filter}")

t_pred = times[-1] + args.horizon
truth_pred = truth(t_pred)


def make_bank(threads):
    cfg = fb.TrackerConfig()
    cfg.filter = fb.FilterType.KALMAN if args.filter == "kalman" else fb.FilterType.ABG
    if args.q is not None:
        cfg.q = args.q
    return fb.TrackerBank(cfg, threads=threads)


def collect(bank):
    ps = bank.predict_all(t_pred)
    pid = np.array([p.id for p in ps])
    xyz = np.array([[p.x, p.y, p.z] for p in ps])
    sig = np.array([[p.sx, p.sy, p.sz] for p in ps])
    return pid, xyz, sig


def run(threads, mode):
    bank = make_bank(threads)
    t0 = time.perf_counter()
    rejected = 0
    if mode == "scan":
        for k in range(T):
            rejected += bank.ingest_array(rows[k])
    else:  # one big shuffled batch
        flat = rows.reshape(-1, 8)
        rejected = bank.ingest_array(flat[rng.permutation(len(flat))])
    t_ing = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = collect(bank)
    t_pred_s = time.perf_counter() - t0
    return bank, out, rejected, t_ing, t_pred_s


# ---- 1. correctness + 4. speed -------------------------------------------------------
print(
    f"\n{'threads':>7} {'mode':>6} {'ingest [s]':>11} {'meas/s':>12} "
    f"{'predict_all [ms]':>17} {'rejected':>9}"
)
results = {}
for th in args.threads:
    for mode in ("scan", "batch"):
        bank, out, rej, ti, tp = run(th, mode)
        results[(th, mode)] = out
        print(
            f"{th:>7} {mode:>6} {ti:>11.3f} {total / ti:>12,.0f} "
            f"{tp * 1e3:>17.2f} {rej:>9}"
        )
        assert rej == 0, "unexpected rejections"
        assert len(bank.ids()) == N, "wrong number of tracks"

pid, xyz, sig = results[(args.threads[0], "scan")]
assert np.array_equal(pid, ids), "ids missing or out of order"
err = xyz - truth_pred
err3 = np.linalg.norm(err, axis=1)
print(f"\nForecast error at +{args.horizon:g} s (vs. truth):")
print(
    f"  3D error  mean {err3.mean():.2f} m   median {np.median(err3):.2f} m   "
    f"p95 {np.percentile(err3, 95):.2f} m   max {err3.max():.2f} m"
)
if np.all(np.isfinite(sig)):
    z = err / sig
    print(
        f"  sigma calibration: mean(z^2) = {np.mean(z**2):.2f} (1.0 = ideal), "
        f"within 3 sigma: {np.mean(np.abs(z) < 3) * 100:.1f}% (ideal 99.7%)"
    )

dm = np.array([bank.diameter(int(i)) for i in ids])
print(
    f"  diameter: mean |rel. error| {np.mean(np.abs(dm - diam) / diam) * 100:.2f}% "
    f"(single-report sigma is {SIG_D * 100:.0f}%)"
)

# ---- 2. determinism across thread counts, 3. scan vs batch ----------------------------
ref = results[(args.threads[0], "scan")][1]
ok = True
for (th, mode), (p2, x2, _) in results.items():
    same = np.array_equal(p2, ids) and np.allclose(x2, ref, rtol=1e-9, atol=1e-6)
    if not same:
        ok = False
        print(
            f"  MISMATCH vs reference: threads={th} mode={mode} "
            f"(max diff {np.abs(x2 - ref).max():.3g} m)"
        )
print("\nDeterminism (threads x ingest mode):", "PASS" if ok else "FAIL")

# ---- 5. edge cases ---------------------------------------------------------------------
print("\nEdge cases:")
b = make_bank(2)
good = rows[0, :3].copy()
b.ingest_array(good)


def expect(name, fn, want_exc=None):
    try:
        r = fn()
        print(
            f"  {name:<28} -> {r!r}"
            if want_exc is None
            else f"  {name:<28} -> no error (FAIL)"
        )
    except Exception as e:  # noqa: BLE001
        status = "ok" if want_exc and isinstance(e, want_exc) else "UNEXPECTED"
        print(f"  {name:<28} -> {type(e).__name__}: {e} [{status}]")


dup = good.copy()
expect("duplicate timestamp", lambda: f"rejected={b.ingest_array(dup)} (want 3)")
bad = rows[1, :3].copy()
bad[0, 2] = np.nan
expect("NaN position", lambda: f"rejected={b.ingest_array(bad)} (want 1)")
expect("predict in the past", lambda: b.predict(0, -10.0), Exception)
expect("unknown id", lambda: b.predict(10**9, 5.0), Exception)
expect("wrong shape", lambda: b.ingest_array(np.zeros((3, 7))), ValueError)
expect("empty array", lambda: f"rejected={b.ingest_array(np.zeros((0, 8)))}")
b.reset()
expect("predict after reset", lambda: len(b.predict_all(5.0)))
