#pragma once
// Single-axis tracking filters sharing one interface, so they can be swapped freely.
// No GUI / plotting dependencies: include this wherever you need the maths.
//
// Usage (per axis):
//   auto f = std::make_unique<KalmanFilter>(x0, v0, a0, q, r_std, p0);   // or AbgFilter
//   f->predict(dt);
//   f->update(dt, f->innovation(measurement));
//   // history of estimates: f->x_k, f->v_k, f->a_k  (one entry per step, plus the initial one)

#include "glm/ext/quaternion_geometric.hpp"
#include "glm/ext/scalar_constants.hpp"
#include "glm/geometric.hpp"

#include <algorithm>
#include <array>
#include <ctime>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <sys/stat.h>
#include <vector>

#include "collision.hpp"
#include "polyfit.hpp"

enum class FilterType : int { ABG = 0, KALMAN = 1, POLYFIT = 2 };
enum class SignalType : int { SINE = 0, BODIES = 1 };

struct Body {
    float x0, y0;
    float vx, vy;
    float ax, ay;
    // appended (keeps the SliderFloat2(&B.x0 / &B.vx / &B.ax) pairs valid)
    float z0 = 0, vz = 0, az = 0;
    float diameter = 5000.0f; // true average diameter [m]

    static constexpr int MAX_BODIES = 500;

    static constexpr float RADIUS = 11100.0f;
    static constexpr float MU = 707500.0f;
    static constexpr float VERT_OFFSET = 500.0f;
    static constexpr float PHOBOS_GRAVITY = 0.0057f;
    static constexpr glm::vec3 PHOBOS_CENTRE = glm::vec3 {0.0f, 0.0f, 0.0f};
    static constexpr glm::vec3 NORTH = {RADIUS, 0.0f, 0.0f};
    static constexpr glm::vec3 WEST = {0.0f, RADIUS, 0.0f};
    static constexpr glm::vec3 SPREAD_M = {300.0f, 300.0f, 500.0f};
    static constexpr glm::vec3 SHUTTLE_LOC = {0.0f, 0.0f, RADIUS};

    struct ScalingStats {
        float scaleMin, scaleMax, radarDiameter;
    };

    static constexpr std::array<ScalingStats, 3> SCALES = {
        ScalingStats {40.0f, 1000.0f, 0.08f},
        ScalingStats {20.0f, 800.0f, 0.05f},
        ScalingStats {10.0f, 400.0f, 0.01f}
    };

    struct NorthWestUp {
        glm::vec3 north, west, up;
    };

    static auto clampFloat(float val, float outMin, float outMax) -> float {
        return glm::min(glm::max(val, outMin), outMax);
    }

    static auto
    remapFloat(float val, float inMin, float inMax, float outMin, float outMax)
        -> float {
        return clampFloat(
            outMin + (outMax - outMin) * (val - inMin) / (inMax - inMin),
            outMin,
            outMax
        );
    }

    static auto clampVector(glm::vec3 val, glm::vec3 outMin, glm::vec3 outMax)
        -> glm::vec3 {
        return glm::min(glm::max(val, outMin), outMax);
    }

    static auto remapVector(
        glm::vec3 val,
        glm::vec3 inMin,
        glm::vec3 inMax,
        glm::vec3 outMin,
        glm::vec3 outMax
    ) -> glm::vec3 {
        return clampVector(
            outMin + (outMax - outMin) * (val - inMin) / (inMax - inMin),
            outMin,
            outMax
        );
    }

    static auto toNWU(auto loc) -> NorthWestUp {
        auto north = glm::normalize(NORTH - loc);
        auto up = -glm::normalize(PHOBOS_CENTRE - loc);
        auto west = glm::cross(up, north);
        return {north, up, west};
    }

    Body(int seed = (int)std::time(NULL)) {
        this->reset(seed);
    }

    void reset(int seed) {
        constexpr auto locMin = SHUTTLE_LOC - SPREAD_M;
        constexpr auto locMax = SHUTTLE_LOC + SPREAD_M;

        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(0.0, 1.0);
        auto randVec = [](auto dis, auto gen) {
            return glm::vec3 {dis(gen), dis(gen), dis(gen)};
        };
        auto loc_random = randVec(dis, gen);
        auto vel_random = dis(gen);
        auto size_random = dis(gen);
        auto azimuth = remapFloat(
            static_cast<float>(dis(gen)),
            0.0,
            1.0,
            0.0,
            2.0 * glm::pi<float>()
        );

        auto loc = remapVector(
            loc_random,
            glm::vec3 {0},
            glm::vec3(1),
            locMin,
            locMax
        );
        auto NWU = toNWU(loc);

        auto vel_mag_max = std::sqrt(MU / RADIUS);
        auto vel_mag = remapFloat(
            vel_random,
            0.0f,
            1.0f,
            0.05f * vel_mag_max,
            0.1f * vel_mag_max
        );
        auto vel = (float)vel_mag * (NWU.north * (float)glm::cos(azimuth))
            - NWU.west * (float)glm::sin(azimuth);
        auto grav = (float)-PHOBOS_GRAVITY * NWU.up;
        auto scaleStats = SCALES[static_cast<int>(dis(gen) * 3)];

        this->x0 = loc.x;
        this->y0 = loc.y;
        this->z0 = loc.z;

        this->vx = vel.x;
        this->vy = vel.y;
        this->vz = vel.z;

        this->ax = grav.x;
        this->ay = grav.y;
        this->az = grav.z;

        this->diameter = (float)remapFloat(
            size_random,
            0.0,
            1.0,
            scaleStats.scaleMin,
            scaleStats.scaleMax
        );
    }
};

struct Params {
    SignalType signal = SignalType::BODIES;
    int n_bodies = 3;
    std::array<Body, Body::MAX_BODIES> bodies {};

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

    float target = 0.0f;

    // sensor spec (1-sigma)
    bool use_sensor = true; // true: x/y/z noise = sig_pos instead of `noise`
    float sig_pos = 2.0f; // [x y z] in Phobos frame [m]
    float sig_rc = 2.0f; // closest-point distance [m]
    float sig_vr = 0.02f; // range rate [m/s]
    float sig_diam_frac = 0.05f; // fraction of true diameter

    void seed_bodies() {
        int body_seed = this->seed + 1000;
        for (int i = 0; i < this->n_bodies; i++) {
            this->bodies.at(i).reset(body_seed + i);
        }
    }
};

inline void clamp_params(Params& p) {
    p.dt = std::max(p.dt, 0.001f);
    p.duration = std::clamp(p.duration, 0.01f, p.dt * 100000.0f);
    p.meas_std = std::max(p.meas_std, 1e-4f);
    p.p0 = std::max(p.p0, 1e-6f);
    p.q = std::max(p.q, 0.0f);
    p.noise = std::max(p.noise, 0.0f);
    p.horizon = std::max(p.horizon, 0.0f);
    p.max_horizon = std::max(p.max_horizon, 0.01f);
    p.warmup = std::max(p.warmup, 0.0f);
    p.n_bodies = std::clamp(p.n_bodies, 1, Body::MAX_BODIES);
    if (p.filter != FilterType::ABG && p.filter != FilterType::KALMAN
        && p.filter != FilterType::POLYFIT)
        p.filter = FilterType::ABG;
    if (p.signal != SignalType::SINE && p.signal != SignalType::BODIES)
        p.signal = SignalType::BODIES;
    p.threads = std::clamp(p.threads, 1, std::max(1, static_cast<int>(16)));
    p.target = std::max(p.target, 0.0f);
    p.sig_pos = std::max(p.sig_pos, 0.0f);
    p.sig_rc = std::max(p.sig_rc, 0.0f);
    p.sig_vr = std::max(p.sig_vr, 0.0f);
    p.sig_diam_frac = std::max(p.sig_diam_frac, 0.0f);
}

inline constexpr int K_SWEEP_POINTS = 60;

struct Forecast {
    float x, v, a;
    float var_x; // variance of the predicted position
};

struct FilterBase {
    std::vector<float> x_k; // position estimates
    std::vector<float> v_k; // velocity estimates
    std::vector<float> a_k; // acceleration estimates

    virtual ~FilterBase() = default;

    // Advance the state by dt and push the predicted state onto the history.
    virtual void predict(float dt) = 0;
    // Correct the latest (predicted) state with the innovation r = measurement - prediction.
    virtual void update(float dt, float r) = 0;

    float innovation(float x_m) const {
        return x_m - x_k.back();
    }

    // Look ahead by `horizon` seconds (>= 0) from the latest state without modifying it.
    virtual Forecast forecast(float horizon) const = 0;

    float predict_position(float horizon) const {
        return forecast(horizon).x;
    }

    static std::unique_ptr<FilterBase> make_filter(const Params& p, float x0);

  protected:
    FilterBase(float x0, float v0, float a0) : x_k {x0}, v_k {v0}, a_k {a0} {}
};

// Fixed-gain alpha-beta-gamma filter.
struct AbgFilter final: FilterBase {
    float alpha, beta, gamma;

    AbgFilter(
        float x0,
        float v0,
        float a0,
        float alpha,
        float beta,
        float gamma
    ) :
        FilterBase(x0, v0, a0),
        alpha(alpha),
        beta(beta),
        gamma(gamma) {}

    void predict(float dt) override;
    void update(float dt, float r) override;
    Forecast forecast(float horizon) const override;
};

// Kalman filter, constant-acceleration model, state [x v a], position measured (H = [1 0 0]).
// Process noise is white jerk with spectral density q; measurement noise std is r_std.
//
// Note: glm matrices are column-major, so M[col][row]. The matrices below are written
// column by column (F is the only non-symmetric one, so its layout matters).
struct KalmanFilter final: FilterBase {
    float q; // process noise (jerk) spectral density
    float R; // measurement variance
    glm::mat3 P; // state covariance

    // p0 = initial variance of every state component (larger = trust early measurements more)
    KalmanFilter(float x0, float v0, float a0, float q, float r_std, float p0) :
        FilterBase(x0, v0, a0),
        q(q),
        R(r_std * r_std),
        P(p0) {} // glm::mat3(scalar) builds a diagonal matrix

    void predict(float dt) override;
    void update(float dt, float r) override;
    Forecast forecast(float horizon) const override;
};

struct PolyFitFilter final: FilterBase {
    poly::Polynomial pf = {};
    std::vector<float> times = {};
    float avg_vel = 0.0;

    PolyFitFilter(float x0, float v0, float a0) :
        FilterBase(x0, v0, a0),
        times {0.0} {}

    void predict(float dt) override;
    void update(float dt, float r) override;
    Forecast forecast(float horizon) const override;
};

struct Track {
    // Filters
    std::unique_ptr<FilterBase> fx, fy, fz;

    // noisy measurements (line up with t[1:])
    std::vector<float> zx, zy, zz;
    // true position
    std::vector<float> tx, ty, tz;
    // true x-y path extended past the end of the run, up to the look-ahead time
    std::vector<float> tx_ext, ty_ext, tz_ext;
    // true velocity / acceleration
    std::vector<float> tvx, tvy, tvz, tax, tay, taz;

    // forecast made at each step (pt = t + horizon)
    std::vector<float> pt, px, py, pz;
    // error vs look-ahead (sig_x empty for ABG)
    std::vector<float> rms_x, rms_y, rms_z, sig_x;
    // where this body is predicted to be, `horizon` after the end of the run, and where it really is
    float fut_x = 0, fut_y = 0, fut_z = 0;
    float fut_tx = 0, fut_ty = 0, fut_tz = 0;
    // distance between forecast made at step i and true position at t + horizon
    std::vector<float> err;

    // measured r_c, v_r, d
    std::vector<float> m_rc, m_vr, m_d;
    // true values
    std::vector<float> t_rc, t_vr, t_d;
};

struct Result {
    std::vector<Track> tracks;
    std::vector<float> t; // time of each state (size = steps + 1)
    std::vector<float> sweep_h; // look-ahead axis for the error plot
    std::vector<float>
        mean_err; // err averaged over all bodies, per step (lines up with t[1:])
    std::vector<float>
        run_avg; // running mean of mean_err from warm-up onwards (NaN before)
    float mean_err_scored = 0.0; // run_avg's final value
    float compute_ms = 0.0; // wall-clock time run_sim took
    Params params;
};

Result run_sim(const Params& p);
