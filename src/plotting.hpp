#pragma once

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <cstring>

#include "filters.hpp"
// ---- plotting helpers ---------------------------------------------------------------------

static const ImVec4 kColX {0.30f, 0.62f, 0.95f, 1.0f}; // x axis: blue

static const ImVec4 kColY {1.00f, 0.55f, 0.20f, 1.0f}; // y axis: orange
static const ImVec4 kColPath {0.30f, 0.85f, 0.50f, 1.0f}; // x-y estimate: green
static const ImVec4
    kColActual {0.90f, 0.90f, 0.90f, 0.75f}; // actual: thin light line
static const ImVec4
    kColMeas {0.65f, 0.65f, 0.65f, 1.0f}; // measurements: grey crosses
static const ImVec4 kColFcast {0.95f, 0.35f, 0.65f, 1.0f}; // forecast: pink

enum PlotId {
    PLOT_XY,
    PLOT_POS,
    PLOT_VEL,
    PLOT_ACC,
    PLOT_ERR,
    PLOT_AVG,
    PLOT_COUNT
};

static const char* kPlotTitle[PLOT_COUNT] = {
    "x-y path",
    "Position",
    "Velocity",
    "Acceleration",
    "Forecast error vs look-ahead",
    "Average forecast accuracy"
};

struct AppState {
    Params p;
    bool show_xy = true, show_pos = true, show_vel = true, show_acc = true;
    bool show_err = true, show_avg = true, auto_run = false;
    bool show_body[Body::MAX_BODIES];
    std::vector<int>
        order {PLOT_XY, PLOT_POS, PLOT_VEL, PLOT_ACC, PLOT_ERR, PLOT_AVG};

    AppState() {
        std::fill_n(show_body, Body::MAX_BODIES, true);
    }
};

// One list of (key, variable) pairs, shared by the reader and the writer.
template<class F>
static void visit_fields(AppState& s, F&& f) {
    Params& p = s.p;
    f("filter", reinterpret_cast<int&>(p.filter));
    f("signal", reinterpret_cast<int&>(p.signal));
    f("n_bodies", p.n_bodies);
    f("alpha", p.alpha);
    f("beta", p.beta);
    f("gamma", p.gamma);
    f("q", p.q);
    f("meas_std", p.meas_std);
    f("p0", p.p0);
    f("amp_x", p.amp_x);
    f("amp_y", p.amp_y);
    f("omega_x", p.omega_x);
    f("omega_y", p.omega_y);
    f("phase_x", p.phase_x);
    f("phase_y", p.phase_y);
    f("noise", p.noise);
    f("seed", p.seed);
    f("dt", p.dt);
    f("v0", p.v0);
    f("duration", p.duration);
    f("horizon", p.horizon);
    f("max_horizon", p.max_horizon);
    f("warmup", p.warmup);
    f("show_xy", s.show_xy);
    f("show_pos", s.show_pos);
    f("show_vel", s.show_vel);
    f("show_acc", s.show_acc);
    f("show_err", s.show_err);
    f("show_avg", s.show_avg);
    f("auto_run", s.auto_run);
    f("threads", p.threads);
    f("target", p.target);
    f("use_sensor", p.use_sensor);
    f("sig_pos", p.sig_pos);
    f("sig_rc", p.sig_rc);
    f("sig_vr", p.sig_vr);
    f("sig_diam_frac", p.sig_diam_frac);
}

static void put(ImGuiTextBuffer* o, const char* k, float v) {
    o->appendf("%s=%.9g\n", k, v);
}

static void put(ImGuiTextBuffer* o, const char* k, int v) {
    o->appendf("%s=%d\n", k, v);
}

static void put(ImGuiTextBuffer* o, const char* k, bool v) {
    o->appendf("%s=%d\n", k, v ? 1 : 0);
}

static void get(const char* s, float& v) {
    v = std::strtof(s, nullptr);
}

static void get(const char* s, int& v) {
    v = std::atoi(s);
}

static void get(const char* s, bool& v) {
    v = std::atoi(s) != 0;
}

// -- moved to filters.hpp ---
// inline void clamp_params(Params& p) {
//     p.dt = std::max(p.dt, 0.001f);
//     p.duration = std::clamp(p.duration, 0.01f, p.dt * 100000.0f);
//     p.meas_std = std::max(p.meas_std, 1e-4f);
//     p.p0 = std::max(p.p0, 1e-6f);
//     p.q = std::max(p.q, 0.0f);
//     p.noise = std::max(p.noise, 0.0f);
//     p.horizon = std::max(p.horizon, 0.0f);
//     p.max_horizon = std::max(p.max_horizon, 0.01f);
//     p.warmup = std::max(p.warmup, 0.0f);
//     p.n_bodies = std::clamp(p.n_bodies, 1, Body::MAX_BODIES);
//     if (p.filter != FilterType::ABG && p.filter != FilterType::KALMAN)
//         p.filter = FilterType::ABG;
//     if (p.signal != SignalType::SINE && p.signal != SignalType::BODIES)
//         p.signal = SignalType::BODIES;
//     p.threads = std::clamp(p.threads, 1, std::max(1, static_cast<int>(16)));
//     p.target = std::max(p.target, 0.0f);
//     p.sig_pos = std::max(p.sig_pos, 0.0f);
//     p.sig_rc = std::max(p.sig_rc, 0.0f);
//     p.sig_vr = std::max(p.sig_vr, 0.0f);
//     p.sig_diam_frac = std::max(p.sig_diam_frac, 0.0f);
// }

// Fall back to safe values if the file was edited by hand or comes from an older version.
static void sanitize(AppState& s) {
    clamp_params(s.p);
    std::vector<int> sorted = s.order;
    std::sort(sorted.begin(), sorted.end());
    bool ok = static_cast<int>(sorted.size()) == PLOT_COUNT;
    for (int i = 0; ok && i < PLOT_COUNT; ++i)
        ok = sorted[i] == i;
    if (!ok)
        s.order = AppState().order;
}

static void*
settings_open(ImGuiContext*, ImGuiSettingsHandler* h, const char* name) {
    return std::strcmp(name, "State") == 0 ? h->UserData : nullptr;
}

static void settings_line(
    ImGuiContext*,
    ImGuiSettingsHandler*,
    void* entry,
    const char* line
) {
    AppState& s = *static_cast<AppState*>(entry);
    const char* eq = std::strchr(line, '=');
    if (!eq)
        return;
    const std::string key(line, eq - line);
    const char* val = eq + 1;

    if (key.rfind("bodyx", 0) == 0 && key.size() > 5 && std::isdigit(key[5])) {
        const int b = std::atoi(key.c_str() + 5);
        if (b >= 0 && b < Body::MAX_BODIES) {
            Body& B = s.p.bodies[b];
            std::sscanf(val, "%f,%f,%f,%f", &B.z0, &B.vz, &B.az, &B.diameter);
        }
        return;
    }

    if (key.rfind("body", 0) == 0 && key.size() > 4 && std::isdigit(key[4])) {
        const int b = std::atoi(key.c_str() + 4);
        if (b >= 0 && b < Body::MAX_BODIES) {
            Body& B = s.p.bodies[b];
            int shown = 1;
            if (std::sscanf(
                    val,
                    "%f,%f,%f,%f,%f,%f,%d",
                    &B.x0,
                    &B.y0,
                    &B.vx,
                    &B.vy,
                    &B.ax,
                    &B.ay,
                    &shown
                )
                == 7)
                s.show_body[b] = shown != 0;
        }
        return;
    }
    if (key == "order") {
        std::vector<int> o;
        for (const char* c = val; *c;) {
            char* end;
            o.push_back(static_cast<int>(std::strtol(c, &end, 10)));
            c = *end == ',' ? end + 1 : end;
            if (end == c && *c)
                break;
        }
        s.order = o;
        return;
    }
    visit_fields(s, [&](const char* k, auto& v) {
        if (key == k)
            get(val, v);
    });
}

static void
settings_write(ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* out) {
    AppState& s = *static_cast<AppState*>(h->UserData);
    out->appendf("[%s][State]\n", h->TypeName);
    visit_fields(s, [&](const char* k, auto& v) { put(out, k, v); });
    for (int b = 0; b < Body::MAX_BODIES; ++b) {
        const Body& B = s.p.bodies[b];
        out->appendf(
            "body%d=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%d\n",
            b,
            B.x0,
            B.y0,
            B.vx,
            B.vy,
            B.ax,
            B.ay,
            s.show_body[b] ? 1 : 0
        );
        out->appendf(
            "bodyx%d=%.9g,%.9g,%.9g,%.9g\n",
            b,
            B.z0,
            B.vz,
            B.az,
            B.diameter
        );
    }
    out->appendf("order=");
    for (size_t i = 0; i < s.order.size(); ++i)
        out->appendf("%s%d", i ? "," : "", s.order[i]);
    out->append("\n\n");
}

struct Decimated {
    std::vector<double> x, y;
};

// xs must be monotonic increasing. Keeps min & max of each pixel bucket,
// in original order, so the drawn line looks the same as the full one.
static void decimate_minmax(
    const double* xs,
    const double* ys,
    int n,
    double x_min,
    double x_max,
    int pixels,
    Decimated& out
) {
    out.x.clear();
    out.y.clear();

    // visible index range (binary search, with one point of margin each side)
    int lo = (int)(std::lower_bound(xs, xs + n, x_min) - xs);
    int hi = (int)(std::upper_bound(xs, xs + n, x_max) - xs);
    lo = std::max(0, lo - 1);
    hi = std::min(n, hi + 1);
    const int cnt = hi - lo;

    if (cnt <= pixels * 2 || pixels <= 0) { // already sparse enough
        out.x.assign(xs + lo, xs + hi);
        out.y.assign(ys + lo, ys + hi);
        return;
    }

    out.x.reserve(pixels * 2 + 2);
    out.y.reserve(pixels * 2 + 2);
    const double per = (double)cnt / pixels;

    for (int px = 0; px < pixels; ++px) {
        const int a = lo + (int)(px * per);
        const int b = std::min(hi, lo + (int)((px + 1) * per));
        if (a >= b)
            continue;

        int imin = a, imax = a;
        for (int i = a + 1; i < b; ++i) {
            if (ys[i] < ys[imin])
                imin = i;
            if (ys[i] > ys[imax])
                imax = i;
        }
        const int first = std::min(imin, imax), second = std::max(imin, imax);
        out.x.push_back(xs[first]);
        out.y.push_back(ys[first]);
        if (second != first) {
            out.x.push_back(xs[second]);
            out.y.push_back(ys[second]);
        }
    }
}
