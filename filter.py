import sys

sys.path.insert(0, "/home/Kieran/UNI/SpaceTeamsPRO/stp-cpp/build")

import numpy as np, filter_bindings as fb

cfg = fb.TrackerConfig()
cfg.filter = fb.FilterType.KALMAN
bank = fb.TrackerBank(cfg, threads=4)

# columns: t, x, y, z, rc, vr, d
scan0 = np.array(
    [
        [0.0, 100.0, 0.0, 0.0, 95.0, 0.0, 10.0],
        [0.0, 5000.0, 200.0, 0.0, 0.0, 0.0, 10.0],
    ]
)


def with_ids(scan):
    # prepend an id column (row order = object id, test only)
    return np.column_stack([np.arange(len(scan)), scan])


ids = list(range(len(scan0)))
print("rejected:", bank.ingest_array(with_ids(scan0)))

scan1 = scan0.copy()
scan1[:, 0] = 1.0
scan1[0, 1] += 1.0
print("rejected:", bank.ingest_array(with_ids(scan1)))

p = bank.predict(ids[0], 5.0)
print(p.x, p.y, p.z, p.sx)
print([(q.id, q.x) for q in bank.predict_all(5.0)])
