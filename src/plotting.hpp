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
    if (p.filter != FilterType::ABG && p.filter != FilterType::KALMAN)
        p.filter = FilterType::ABG;
    if (p.signal != SignalType::SINE && p.signal != SignalType::BODIES)
        p.signal = SignalType::BODIES;
    p.threads = std::clamp(p.threads, 1, std::max(1, static_cast<int>(16)));
}

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
    }
    out->appendf("order=");
    for (size_t i = 0; i < s.order.size(); ++i)
        out->appendf("%s%d", i ? "," : "", s.order[i]);
    out->append("\n\n");
}
