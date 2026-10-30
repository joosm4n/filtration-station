#include "filters.hpp" // FilterBase, AbgFilter, KalmanFilter

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace {

struct Truth {
    double x, y, z, vx, vy, vz, ax, ay, az;
};

Truth truth_at(const Params& p, int b, double t) {
    if (p.signal == SignalType::SINE) {
        const double wx = p.omega_x, wy = p.omega_y;
        const double sx = std::sin(wx * t + p.phase_x),
                     cx = std::cos(wx * t + p.phase_x);
        const double sy = std::sin(wy * t + p.phase_y),
                     cy = std::cos(wy * t + p.phase_y);
        return {
            p.amp_x * sx,
            p.amp_y * sy,
            0.0,
            p.amp_x * wx * cx,
            p.amp_y * wy * cy,
            0.0,
            -p.amp_x * wx * wx * sx,
            -p.amp_y * wy * wy * sy,
            0.0,
        };
    }
    const Body& B = p.bodies[b];
    return {
        B.x0 + B.vx * t + 0.5 * B.ax * t * t,
        B.y0 + B.vy * t + 0.5 * B.ay * t * t,
        B.z0 + B.vz * t + 0.5 * B.az * t * t,
        B.vx + B.ax * t,
        B.vy + B.ay * t,
        B.vz + B.az * t,
        B.ax,
        B.ay,
        B.az
    };
}

// Run fn(0..n-1) on up to `max_threads` threads. The calling thread is one of them,
// so max_threads = 1 means no extra threads are created.
void parallel_for(int n, int max_threads, const std::function<void(int)>& fn) {
    const int nthreads = std::clamp(max_threads, 1, std::max(n, 1));
    if (nthreads == 1) {
        for (int i = 0; i < n; ++i)
            fn(i);
        return;
    }
    std::atomic<int> next {0};
    auto work = [&] {
        for (int i; (i = next.fetch_add(1)) < n;)
            fn(i);
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (int i = 1; i < nthreads; ++i)
        pool.emplace_back(work);
    work();
    for (auto& th : pool)
        th.join();
}

// Everything for one body. Touches only its own Track, so bodies can run concurrently.
void run_body(const Params& p, int b, int steps, Track& k) {
    const Truth t0 = truth_at(p, b, 0.0);
    k.fx = FilterBase::make_filter(p, t0.x);
    k.fy = FilterBase::make_filter(p, t0.y);
    k.fz = FilterBase::make_filter(p, t0.z);

    std::mt19937 rng(static_cast<unsigned>(p.seed + b));

    for (auto* v :
         {&k.zx,   &k.zy,   &k.zz,  &k.tx,   &k.ty,   &k.tz,  &k.tvx, &k.tvy,
          &k.tvz,  &k.tax,  &k.tay, &k.taz,  &k.pt,   &k.px,  &k.py,  &k.pz,
          &k.m_rc, &k.m_vr, &k.m_d, &k.t_rc, &k.t_vr, &k.t_d, &k.err})
        v->reserve(steps);

    std::vector<double> se_x(K_SWEEP_POINTS, 0.0), se_y(K_SWEEP_POINTS, 0.0),
        se_z(K_SWEEP_POINTS, 0.0);
    std::vector<double> sg_x(K_SWEEP_POINTS, 0.0);
    int scored = 0;

    for (int i = 0; i < steps; ++i) {
        const double t = (i + 1) * p.dt;

        k.fx->predict(p.dt);
        k.fy->predict(p.dt);
        k.fz->predict(p.dt);

        const Truth a = truth_at(p, b, t);
        k.tx.push_back(a.x);
        k.ty.push_back(a.y);
        k.tz.push_back(a.z);
        k.tvx.push_back(a.vx);
        k.tvy.push_back(a.vy);
        k.tvz.push_back(a.vz);
        k.tax.push_back(a.ax);
        k.tay.push_back(a.ay);
        k.taz.push_back(a.az);

        const double sp = p.use_sensor ? p.sig_pos : p.noise;
        std::normal_distribution<double> gp(0.0, std::max(sp, 1e-9));
        std::normal_distribution<double> g_rc(
            0.0,
            std::max<double>(p.sig_rc, 1e-9)
        );
        std::normal_distribution<double> g_vr(
            0.0,
            std::max<double>(p.sig_vr, 1e-9)
        );
        std::normal_distribution<double> g_d(
            0.0,
            std::max<double>(p.sig_diam_frac, 1e-9)
        );

        double zx = a.x, zy = a.y, zz = a.z;
        if (sp > 0.0f) {
            zx += gp(rng);
            zy += gp(rng);
            zz += gp(rng);
        }

        const double D =
            (p.signal == SignalType::BODIES) ? p.bodies[b].diameter : 0.0;
        const double rho =
            std::max(std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-9);
        const double rc = rho - 0.5 * D;
        const double vr = (a.x * a.vx + a.y * a.vy + a.z * a.vz) / rho;

        k.t_rc.push_back(rc);
        k.t_vr.push_back(vr);
        k.t_d.push_back(D);
        k.m_rc.push_back(rc + g_rc(rng));
        k.m_vr.push_back(vr + g_vr(rng));
        k.m_d.push_back(D * (1.0 + g_d(rng)));

        k.zx.push_back(zx);
        k.zy.push_back(zy);
        k.zz.push_back(zz);
        k.fx->update(p.dt, k.fx->innovation(zx));
        k.fy->update(p.dt, k.fy->innovation(zy));
        k.fz->update(p.dt, k.fz->innovation(zz));

        const double h = p.target > 0.0f ? std::max(0.0, p.target - t)
                                         : static_cast<double>(p.horizon);

        const Forecast fx = k.fx->forecast(h);
        const Forecast fy = k.fy->forecast(h);
        const Forecast fz = k.fz->forecast(h);
        k.pt.push_back(t + h);
        k.px.push_back(fx.x);
        k.py.push_back(fy.x);
        k.pz.push_back(fz.x);

        const Truth ft = truth_at(p, b, t + p.horizon);
        k.err.push_back(std::hypot(fx.x - ft.x, fy.x - ft.y, fz.x - ft.z));

        if (t >= p.warmup) {
            ++scored;
            for (int j = 0; j < K_SWEEP_POINTS; ++j) {
                const double h2 = p.max_horizon * (j + 1) / K_SWEEP_POINTS;
                const Forecast ax = k.fx->forecast(h2), ay = k.fy->forecast(h2),
                               az = k.fz->forecast(h2);
                const Truth f = truth_at(p, b, t + h2);
                const double ex = ax.x - f.x, ey = ay.x - f.y, ez = az.x - f.z;
                se_x[j] += ex * ex;
                se_y[j] += ey * ey;
                se_z[j] += ez * ez;
                sg_x[j] += std::sqrt(ax.var_x); // NaN for ABG, unused then
            }
        }
    }

    const bool has_sigma = p.filter == FilterType::KALMAN;
    k.rms_x.assign(K_SWEEP_POINTS, 0.0);
    k.rms_y.assign(K_SWEEP_POINTS, 0.0);
    if (has_sigma)
        k.sig_x.assign(K_SWEEP_POINTS, 0.0);
    if (scored > 0) {
        for (int j = 0; j < K_SWEEP_POINTS; ++j) {
            k.rms_x[j] = std::sqrt(se_x[j] / scored);
            k.rms_y[j] = std::sqrt(se_y[j] / scored);
            if (has_sigma)
                k.sig_x[j] = sg_x[j] / scored;
        }
    }

    const double t_end = steps * p.dt;
    k.fut_x = k.px.back();
    k.fut_y = k.py.back();
    k.fut_z = k.pz.back();
    const double h_end = p.target > 0.0f ? std::max(0.0, p.target - t_end)
                                         : static_cast<double>(p.horizon);
    const Truth f = truth_at(p, b, t_end + h_end);
    k.fut_tx = f.x;
    k.fut_ty = f.y;
    k.fut_tz = f.z;

    // true path continued from the end of the run to the look-ahead time
    k.tx_ext = k.tx;
    k.ty_ext = k.ty;
    k.tz_ext = k.tz;
    for (double te = t_end + p.dt; te < t_end + h_end - 1e-9; te += p.dt) {
        const Truth e = truth_at(p, b, te);
        k.tx_ext.push_back(e.x);
        k.ty_ext.push_back(e.y);
        k.tz_ext.push_back(e.z);
    }
    if (h_end > 0.0) {
        k.tx_ext.push_back(f.x);
        k.ty_ext.push_back(f.y);
        k.tz_ext.push_back(f.z);
    }
}

glm::dmat3 make_F(double dt) {
    const double dt2 = dt * dt;
    return glm::dmat3 {
        glm::dvec3 {1.0, 0.0, 0.0},
        glm::dvec3 {dt, 1.0, 0.0},
        glm::dvec3 {0.5 * dt2, dt, 1.0}
    };
}

glm::dmat3 make_Q(double dt, double q) {
    const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt, dt5 = dt4 * dt;
    return q
        * glm::dmat3 {
            glm::dvec3 {dt5 / 20.0, dt4 / 8.0, dt3 / 6.0},
            glm::dvec3 {dt4 / 8.0, dt3 / 3.0, dt2 / 2.0},
            glm::dvec3 {dt3 / 6.0, dt2 / 2.0, dt}
        };
}
} // namespace

void AbgFilter::predict(double dt) {
    x_k.push_back(x_k.back() + v_k.back() * dt + 0.5 * a_k.back() * dt * dt);
    v_k.push_back(v_k.back() + a_k.back() * dt);
    a_k.push_back(a_k.back());
}

void AbgFilter::update(double dt, double r) {
    x_k.back() += alpha * r;
    v_k.back() += beta * r / dt;
    a_k.back() += (2 * gamma * r) / (dt * dt);
}

void KalmanFilter::predict(double dt) {
    const glm::dmat3 F = make_F(dt);

    const glm::dvec3 s = F * glm::dvec3 {x_k.back(), v_k.back(), a_k.back()};
    x_k.push_back(s.x);
    v_k.push_back(s.y);
    a_k.push_back(s.z);

    P = F * P * glm::transpose(F) + make_Q(dt, q);
}

void KalmanFilter::update(double /*dt*/, double r) {
    const double S = P[0][0] + R; // innovation variance (H P H^T + R)
    const glm::dvec3 K = P[0] / S; // Kalman gain: column 0 of P (= P H^T)

    const glm::dvec3 s =
        glm::dvec3 {x_k.back(), v_k.back(), a_k.back()} + K * r;
    x_k.back() = s.x;
    v_k.back() = s.y;
    a_k.back() = s.z;

    // P = (I - K H) P, where K H has K as its first column and zeros elsewhere
    glm::dmat3 KH {0.0};
    KH[0] = K;
    const glm::dmat3 Pn = (glm::dmat3 {1.0} - KH) * P;

    // symmetrise to limit numerical drift
    P = 0.5 * (Pn + glm::transpose(Pn));
}

Forecast KalmanFilter::forecast(double h) const {
    const glm::dmat3 F = make_F(h);
    const glm::dvec3 s = F * glm::dvec3 {x_k.back(), v_k.back(), a_k.back()};
    const glm::dmat3 Pf = F * P * glm::transpose(F) + make_Q(h, q);
    return {s.x, s.y, s.z, Pf[0][0]};
}

// x0 = initial position estimate (we start on the actual signal's value at t = 0)
std::unique_ptr<FilterBase>
FilterBase::make_filter(const Params& p, double x0) {
    const double v0 = static_cast<double>(p.v0);
    if (p.filter == FilterType::KALMAN) {
        return std::make_unique<KalmanFilter>(
            x0,
            v0,
            0.0,
            p.q,
            p.meas_std,
            p.p0
        );
    }
    return std::make_unique<AbgFilter>(x0, v0, 0.0, p.alpha, p.beta, p.gamma);
}

Forecast AbgFilter::forecast(double h) const {
    const double x = x_k.back(), v = v_k.back(), a = a_k.back();
    return {
        x + v * h + 0.5 * a * h * h,
        v + a * h,
        a,
        std::numeric_limits<double>::quiet_NaN()
    };
}

// Run the whole simulation from t = 0 to t = duration.
Result run_sim(const Params& p) {
    const auto t_start = std::chrono::steady_clock::now();

    Result res;
    res.params = p;
    const int nb = p.signal == SignalType::SINE
        ? 1
        : std::clamp(p.n_bodies, 1, Body::MAX_BODIES);
    const int steps =
        std::max(1, static_cast<int>(std::lround(p.duration / p.dt)));

    res.tracks.resize(nb);
    parallel_for(nb, p.threads, [&](int b) {
        run_body(p, b, steps, res.tracks[b]);
    });

    // combine across bodies (cheap, so serial)
    res.mean_err.assign(steps, 0.0);
    res.run_avg.assign(steps, std::numeric_limits<double>::quiet_NaN());
    double acc = 0.0;
    int cnt = 0;
    for (int i = 0; i < steps; ++i) {
        double s = 0.0;
        for (const Track& k : res.tracks)
            s += k.err[i];
        res.mean_err[i] = s / nb;
        if ((i + 1) * p.dt >= p.warmup) {
            acc += res.mean_err[i];
            ++cnt;
            res.run_avg[i] = acc / cnt;
        }
    }
    res.mean_err_scored = cnt > 0 ? acc / cnt : 0.0;

    res.sweep_h.resize(K_SWEEP_POINTS);
    for (int j = 0; j < K_SWEEP_POINTS; ++j)
        res.sweep_h[j] = p.max_horizon * (j + 1) / K_SWEEP_POINTS;

    res.t.resize(static_cast<std::size_t>(steps) + 1);
    for (std::size_t i = 0; i < res.t.size(); ++i)
        res.t[i] = static_cast<double>(i) * p.dt;

    res.compute_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_start
    )
                         .count();
    return res;
}
