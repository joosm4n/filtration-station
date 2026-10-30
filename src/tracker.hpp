#pragma once
// Online tracking API on top of filters.hpp.
//
//   TrackerBank bank(cfg, /*threads=*/4);
//   bank.ingest(measurements);                 // any number of objects, any order
//   Prediction p = bank.predict(id, t_future); // position (and velocity) at time t
//
// Design rules (so it can be wrapped for Python / used from many threads):
//   * public types are plain structs of double / int64 / enum, no glm, no unique_ptr
//   * Tracker      : one object, NOT thread-safe on its own
//   * TrackerBank  : thread-safe (shared_mutex); ingest() runs objects in parallel
//   * nothing throws inside worker threads; errors are thrown on the calling thread
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "filters.hpp"

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// One sensor report for one object.
struct Measurement {
    int64_t id = 0; // which object this report belongs to
    double t = 0.0; // measurement time since start of simulation [s] (exact)
    double x = 0.0, y = 0.0, z = 0.0; // centre of mass, Phobos-fixed frame [m]
    double rc = kNaN; // closest-point distance from shuttle [m]
    double vr = kNaN; // range rate [m/s]
    double d = kNaN; // average diameter [m]
};

// 1-sigma sensor accuracy. Defaults match the spec table.
struct SensorSpec {
    double sig_pos = 2.0; // [x y z] [m]
    double sig_rc = 2.0; // [m]
    double sig_vr = 0.02; // [m/s]
    double sig_diam_frac = 0.05; // fraction of true diameter
};

struct TrackerConfig {
    FilterType filter = FilterType::KALMAN;
    // alpha-beta-gamma gains
    double alpha = 0.5, beta = 0.02, gamma = 0.01;
    // Kalman tuning (measurement noise comes from sensor.sig_pos)
    double q = 1.0; // process noise (jerk spectral density)
    double p0 = 1.0; // initial state variance
    double v0 = 0.0; // initial velocity guess (per axis)
    SensorSpec sensor;
};

struct Prediction {
    int64_t id = 0;
    double t = 0.0; // time the prediction is for
    double x = 0, y = 0, z = 0; // predicted position [m]
    double vx = 0, vy = 0, vz = 0; // predicted velocity [m/s]
    double sx = kNaN, sy = kNaN, sz = kNaN; // 1-sigma position (NaN for ABG)
};

// Tracks a single object.
class Tracker {
  public:
    explicit Tracker(TrackerConfig cfg = {}) : cfg_(cfg) {}

    // Returns false (and ignores the report) if m.t is not strictly after the
    // previous report, or the position is not finite.
    bool update(const Measurement& m);

    bool initialised() const {
        return static_cast<bool>(fx_);
    }

    double last_time() const {
        return t_last_;
    }

    int n_updates() const {
        return n_;
    }

    // Predicted state at absolute time t (t >= last_time()).
    // Throws std::logic_error if no measurement yet, std::invalid_argument if t is in the past.
    Prediction predict(double t) const;

    // Running mean of the diameter reports (NaN if none) and its 1-sigma uncertainty.
    double diameter() const {
        return d_n_ ? d_sum_ / d_n_ : kNaN;
    }

    double diameter_sigma() const {
        return d_n_
            ? cfg_.sensor.sig_diam_frac * diameter() / std::sqrt(double(d_n_))
            : kNaN;
    }

  private:
    TrackerConfig cfg_;
    std::unique_ptr<FilterBase> fx_, fy_, fz_;
    double t_last_ = 0.0;
    int n_ = 0;
    double d_sum_ = 0.0;
    int d_n_ = 0;
};

// Many objects, keyed by id.
class TrackerBank {
  public:
    explicit TrackerBank(TrackerConfig cfg = {}, int threads = 1) :
        cfg_(cfg),
        threads_(threads < 1 ? 1 : threads) {}

    // Feed a batch of reports (mixed ids / unsorted is fine; they are grouped by id and
    // sorted by time). Objects are processed in parallel. Returns the number rejected.
    int ingest(const std::vector<Measurement>& ms);

    // Throws std::out_of_range for an unknown id (plus the Tracker::predict errors).
    Prediction predict(int64_t id, double t) const;

    // Prediction for every tracked object at time t (parallel). Throws if any object has
    // a report later than t.
    std::vector<Prediction> predict_all(double t) const;

    std::vector<int64_t> ids() const;
    double diameter(int64_t id) const;
    void reset();

    void set_threads(int n) {
        threads_ = n < 1 ? 1 : n;
    }

  private:
    TrackerConfig cfg_;
    int threads_;
    std::map<int64_t, Tracker>
        trackers_; // node-based: element addresses stay valid
    mutable std::shared_mutex mx_;
};
