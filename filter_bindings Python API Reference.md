# filter\_bindings: Python API Reference

Python bindings (pybind11) for the C++ object-tracking library in `tracker.hpp` / `filters.hpp`. Each object is tracked independently with either an alpha-beta-gamma filter or a constant-acceleration Kalman filter, per axis (x, y, z).

## Quick start

```python
import numpy as np
import filter_bindings as pf

cfg = pf.TrackerConfig()
cfg.filter = pf.FilterType.KALMAN
cfg.sensor.sig_pos = 2.0

bank = pf.TrackerBank(cfg, threads=4)

# arr: (N, 8) float64 -> id, t, x, y, z, rc, vr, d
rejected = bank.ingest_array(arr)

p = bank.predict(7, 12.5)       # object 7 at t = 12.5 s
print(p.x, p.y, p.z, p.sx)

all_preds = bank.predict_all(20.0)
```

## Conventions

- **Time** `t` is in seconds since the start of the simulation.
- **Positions** are in metres, in the Phobos-fixed frame (centre of mass).
- **Velocities** are in m/s.
- **Missing values** are `float('nan')` (for `rc`, `vr`, `d`).
- **Object ids** are integers (64-bit on the C++ side).

## Enums

### `FilterType`

| Value | Meaning |
| --- | --- |
| `FilterType.ABG` | Fixed-gain alpha-beta-gamma filter |
| `FilterType.KALMAN` | Kalman filter, constant-acceleration model, state \[x, v, a\] |

## Configuration classes

All fields are read/write attributes. Each class has a default constructor, and defaults are listed below. Constructors do not accept keyword arguments, so set fields after creating the object.

### `SensorSpec`

1-sigma sensor accuracy. Defaults match the spec table.

| Attribute | Type | Default | Description |
| --- | --- | --- | --- |
| `sig_pos` | float | `2.0` | Position (x, y, z) noise \[m\]. Used as the Kalman measurement noise. |
| `sig_rc` | float | `2.0` | Closest-point distance noise \[m\] |
| `sig_vr` | float | `0.02` | Range-rate noise \[m/s\] |
| `sig_diam_frac` | float | `0.05` | Diameter noise as a fraction of the true diameter |

### `TrackerConfig`

| Attribute | Type | Default | Used by | Description |
| --- | --- | --- | --- | --- |
| `filter` | `FilterType` | `KALMAN` | both | Which filter to run |
| `alpha` | float | `0.5` | ABG | Position gain |
| `beta` | float | `0.02` | ABG | Velocity gain |
| `gamma` | float | `0.01` | ABG | Acceleration gain |
| `q` | float | `1.0` | Kalman | Process noise (jerk spectral density) |
| `p0` | float | `1.0` | Kalman | Initial variance of every state component. Larger values trust early measurements more. |
| `v0` | float | `0.0` | both | Initial velocity guess (per axis) |
| `sensor` | `SensorSpec` | defaults above | both | Sensor accuracy |

`cfg.sensor` returns a reference to the nested object, so `cfg.sensor.sig_pos = 3.0` modifies `cfg`. Configure the bank's `TrackerConfig` before constructing it. The bank stores its own copy, so changing `cfg` afterwards has no effect.

## Data classes

### `Measurement`

One sensor report for one object. Fields are read/write.

| Attribute | Type | Default | Description |
| --- | --- | --- | --- |
| `id` | int | `0` | Object this report belongs to |
| `t` | float | `0.0` | Measurement time \[s\] |
| `x`, `y`, `z` | float | `0.0` | Centre of mass position \[m\] |
| `rc` | float | `nan` | Closest-point distance from the shuttle \[m\] |
| `vr` | float | `nan` | Range rate \[m/s\] |
| `d` | float | `nan` | Average diameter \[m\] |

```python
m = pf.Measurement()
m.id, m.t = 7, 1.0
m.x, m.y, m.z = 100.0, -20.0, 5.0
m.d = 5100.0
```

### `Prediction`

Returned by `predict` / `predict_all`. All attributes are **read-only**, and the class cannot be constructed from Python.

| Attribute | Type | Description |
| --- | --- | --- |
| `id` | int | Object id |
| `t` | float | Time the prediction is for \[s\] |
| `x`, `y`, `z` | float | Predicted position \[m\] |
| `vx`, `vy`, `vz` | float | Predicted velocity \[m/s\] |
| `sx`, `sy`, `sz` | float | 1-sigma position uncertainty \[m\]. `nan` for the ABG filter. |

## `TrackerBank`

Tracks many objects keyed by id. The C++ class is thread-safe (guarded by a shared mutex), and `ingest` and `predict_all` process objects in parallel across the configured number of threads.

### `TrackerBank(config=TrackerConfig(), threads=1)`

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `config` | `TrackerConfig` | defaults | Filter and sensor configuration |
| `threads` | int | `1` | Worker threads for batch operations. Values below 1 are treated as 1. |

### `set_threads(n: int) -> None`

Change the number of worker threads. Values below 1 are treated as 1.

### `ingest(measurements: list[Measurement]) -> int`

Feed a batch of reports. Reports may have mixed ids and any order: they are grouped by id and sorted by time, and the objects are processed in parallel. The GIL is released while the batch is processed.

Returns the **number of rejected reports**. A report is rejected (and ignored) if:

- its `t` is not strictly after the previous accepted report for that object, or
- its position is not finite.

Objects are created automatically the first time their id is seen.

### `ingest_array(a: numpy.ndarray) -> int`

Same as `ingest`, but takes a NumPy array and avoids creating one Python object per report, so it is the faster choice for large batches.

- Shape must be `(N, 8)`, otherwise `ValueError` is raised.
- Columns, in order: `id, t, x, y, z, rc, vr, d`.
- Any array-like is accepted. It is converted to C-contiguous `float64` (copied if necessary).
- Use `np.nan` for missing `rc`, `vr`, `d`.
- `id` is stored as a float64, then truncated to an integer, so ids above 2^53 lose precision.
- Returns the number of rejected reports. The GIL is released during processing.

```python
arr = np.array([
    [7, 1.0, 100.0, -20.0, 5.0, np.nan, np.nan, 5100.0],
    [7, 2.0, 101.5, -19.0, 5.2, np.nan, np.nan, np.nan],
])
rejected = bank.ingest_array(arr)
```

### `ingest_radar_array(rows, ids, now, observer_position, observer_velocity, clock_offset=0.0) -> int`

WSO-compatible bulk entry point. `rows` uses the public radar shape `(N, 3, 3)`:

- `rows[:, 0, :]`: relative XYZ position in the Phobos-fixed frame
- `rows[:, 1, :]`: closest range, range rate, diameter
- `rows[:, 2, 0]`: source time; the remaining two values must be zero

`ids` supplies local hypothesis ids from the caller's association layer. The
binding backdates the observer with constant velocity and converts each row to
the absolute XYZ `Measurement` used by `TrackerBank`. `now` and all timestamps
share the same elapsed-seconds clock. This method does not perform anonymous
radar association; use `wso-stm`'s `NativeRadarTracker` for that layer.

### `predict(id: int, t: float) -> Prediction`

Predicted state of one object at absolute time `t` (seconds). Requires ` t >=  ` the object's last report time. The GIL is released during the call.

### `predict_all(t: float) -> list[Prediction]`

Prediction for every tracked object at time `t`, computed in parallel. Raises if any object has a report later than `t`. The GIL is released during the call.

### `ids() -> list[int]`

Ids of all tracked objects.

### `diameter(id: int) -> float`

Running mean of the diameter reports `d` for that object \[m\], or `nan` if none have been received.

### `reset() -> None`

Discard all tracked objects. Configuration and thread count are kept.

## Exceptions

C++ exceptions are translated by pybind11:

| Python exception | Raised when |
| --- | --- |
| `ValueError` | `ingest_array` input is not shape `(N, 8)`. `predict` is asked for a time before the last report. |
| `IndexError` | `predict` is called with an unknown id. |
| `RuntimeError` | An object exists but has no usable measurement yet (`std::logic_error`), or another unexpected C++ error. |

Worker threads never throw, so errors always surface on the calling thread.

## Notes and limitations

- **ABG vs Kalman parameters:** `alpha`, `beta`, `gamma` apply only to `ABG`. `q` and `p0` apply only to `KALMAN`.
- **Uncertainty:** `sx`, `sy`, `sz` are only meaningful with the Kalman filter.
- **Unused sensor fields:** in the code provided, only `sig_pos` (filter noise) and `sig_diam_frac` (diameter uncertainty) are consumed. `rc` and `vr` are accepted but not used by the filters.
- **Not exposed to Python:** `Tracker` (single object), `diameter_sigma`, and the simulation types from `filters.hpp` (`Params`, `Run`, `run_sim`, etc.).
- **No `__repr__`, pickling or type stubs** are defined. A `.pyi` stub would be a worthwhile addition for IDE autocompletion.
- **Time ordering:** reports for an object must have strictly increasing `t`. Duplicate timestamps are rejected.
