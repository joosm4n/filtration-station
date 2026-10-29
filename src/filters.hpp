#pragma once
// Single-axis tracking filters sharing one interface, so they can be swapped freely.
// No GUI / plotting dependencies: include this wherever you need the maths.
//
// Usage (per axis):
//   auto f = std::make_unique<KalmanFilter>(x0, v0, a0, q, r_std, p0);   // or AbgFilter
//   f->predict(dt);
//   f->update(dt, f->innovation(measurement));
//   // history of estimates: f->x_k, f->v_k, f->a_k  (one entry per step, plus the initial one)

#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

enum class FilterType : int { ABG = 0, KALMAN = 1 };
enum class SignalType : int { SINE = 0, BODIES = 1 };

struct Body {
    float x0, y0;
    float vx, vy;
    float ax, ay;

    static constexpr int MAX_BODIES = 100;
};

struct Params {
    SignalType signal = SignalType::BODIES;
    int n_bodies = 3;
    std::array<Body, Body::MAX_BODIES> bodies {{
        Body {0.0, 0.0, 1.0, 0.5, 0.0, -0.5},
        Body {2.0, 1.0, -0.5, 1.0, 0.3, 0.0},
        Body {-1.0, 2.0, 0.8, -0.8, -0.2, 0.4},
    }};

    FilterType filter = FilterType::ABG;

    // alpha-beta-gamma gains (shared by both axes)
    float alpha = 0.5f;
    float beta = 0.02f;
    float gamma = 0.01f;

    // Kalman tuning (shared by both axes)
    float q = 1.0f; // process noise (jerk spectral density)
    float meas_std = 0.1f; // assumed measurement noise std (R = meas_std^2)
    float p0 = 1.0f; // initial state variance
    //
    // actual signal per axis: A * sin(omega * t + B)   (a Lissajous curve in the x-y plane)
    float amp_x = 1.0f; // A
    float amp_y = 1.0f;
    float omega_x = 1.0f; // omega = 1 gives exactly A * sin(B + t)
    float omega_y = 2.0f;
    float phase_x = 0.0f; // B [rad]
    float phase_y = 0.0f;
    float noise = 0.0f; // actual noise std dev, per axis
    int seed = 42;

    // simulation setup
    float dt = 0.1f;
    float v0 = 0.1f; // initial velocity estimate (both axes)
    float duration = 6.0f; // total simulated time [s]

    // prediction
    float horizon = 0.5f; // look-ahead drawn on the plots [s]
    float max_horizon = 3.0f; // right edge of the error-vs-look-ahead plot [s]
    float warmup = 1.0f; // ignore this much time at the start when scoring [s]

    int threads =
        3; // max threads used to run bodies in parallel (including the caller)
};

inline constexpr int K_SWEEP_POINTS = 60;

struct Forecast {
    double x, v, a;
    double var_x; // variance of the predicted position
};

struct FilterBase {
    std::vector<double> x_k; // position estimates
    std::vector<double> v_k; // velocity estimates
    std::vector<double> a_k; // acceleration estimates

    virtual ~FilterBase() = default;

    // Advance the state by dt and push the predicted state onto the history.
    virtual void predict(double dt) = 0;
    // Correct the latest (predicted) state with the innovation r = measurement - prediction.
    virtual void update(double dt, double r) = 0;

    double innovation(double x_m) const {
        return x_m - x_k.back();
    }

    // Look ahead by `horizon` seconds (>= 0) from the latest state without modifying it.
    virtual Forecast forecast(double horizon) const = 0;

    double predict_position(double horizon) const {
        return forecast(horizon).x;
    }

    static std::unique_ptr<FilterBase> make_filter(const Params& p, double x0);

  protected:
    FilterBase(double x0, double v0, double a0) :
        x_k {x0},
        v_k {v0},
        a_k {a0} {}
};

// Fixed-gain alpha-beta-gamma filter.
struct AbgFilter final: FilterBase {
    double alpha, beta, gamma;

    AbgFilter(
        double x0,
        double v0,
        double a0,
        double alpha,
        double beta,
        double gamma
    ) :
        FilterBase(x0, v0, a0),
        alpha(alpha),
        beta(beta),
        gamma(gamma) {}

    void predict(double dt) override;
    void update(double dt, double r) override;
    Forecast forecast(double horizon) const override;
};

// Kalman filter, constant-acceleration model, state [x v a], position measured (H = [1 0 0]).
// Process noise is white jerk with spectral density q; measurement noise std is r_std.
//
// Note: glm matrices are column-major, so M[col][row]. The matrices below are written
// column by column (F is the only non-symmetric one, so its layout matters).
struct KalmanFilter final: FilterBase {
    double q; // process noise (jerk) spectral density
    double R; // measurement variance
    glm::dmat3 P; // state covariance

    // p0 = initial variance of every state component (larger = trust early measurements more)
    KalmanFilter(
        double x0,
        double v0,
        double a0,
        double q,
        double r_std,
        double p0
    ) :
        FilterBase(x0, v0, a0),
        q(q),
        R(r_std * r_std),
        P(p0) {} // glm::dmat3(scalar) builds a diagonal matrix

    void predict(double dt) override;
    void update(double /*dt*/, double r) override;
    Forecast forecast(double horizon) const override;
};

struct Track {
    std::unique_ptr<FilterBase> fx, fy;
    std::vector<double> zx, zy; // noisy measurements (line up with t[1:])
    std::vector<double> tx, ty; // true position
    std::vector<double> tvx, tvy, tax, tay; // true velocity / acceleration
    std::vector<double> pt, px,
        py; // forecast made at each step (pt = t + horizon)
    std::vector<double> rms_x, rms_y,
        sig_x; // error vs look-ahead (sig_x empty for ABG)
    // where this body is predicted to be, `horizon` after the end of the run, and where it really is
    double fut_x = 0, fut_y = 0, fut_tx = 0, fut_ty = 0;
    std::vector<double>
        err; // distance between forecast made at step i and true position at t + horizon
};

struct Result {
    std::vector<Track> tracks;
    std::vector<double> t; // time of each state (size = steps + 1)
    std::vector<double> sweep_h; // look-ahead axis for the error plot
    std::vector<double>
        mean_err; // err averaged over all bodies, per step (lines up with t[1:])
    std::vector<double>
        run_avg; // running mean of mean_err from warm-up onwards (NaN before)
    double mean_err_scored = 0.0; // run_avg's final value
    double compute_ms = 0.0; // wall-clock time run_sim took
    Params params;
};

Result run_sim(const Params& p);
