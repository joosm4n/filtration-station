// 2D tracking filter viewer: alpha-beta-gamma or Kalman (one independent filter per axis).
// Everything is drawn in a single Dear ImGui window: controls on the left, ImPlot plots on
// the right. Whenever a control changes, on auto the filter is re-run from scratch over a fixed time
// span (the "duration" slider); the plots just redraw every frame.

#include "filters.hpp"
#include "imgui.h"
#include "plotting.hpp"
#include "simulation_worker.hpp"
#include "viewer_base.hpp"

#include <GLFW/glfw3.h>
#include <imgui_internal.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <thread>

using SimWorker = SimulationWorker<Params, Result, run_sim>;

static void draw_line(
    const char* label,
    const double* xs,
    const double* ys,
    int n,
    const ImVec4& col,
    bool dots,
    float weight = 1.5f
) {
    if (n <= 0)
        return;
    ImPlotSpec spec; // ImPlot >= 1.0: styling is passed per plot call
    spec.LineColor = col;
    spec.LineWeight = weight;
    if (dots) {
        spec.Marker = ImPlotMarker_Circle;
        spec.MarkerSize = 3.0f;
        spec.MarkerFillColor = col;
        spec.MarkerLineColor = col;
    }
    ImPlot::PlotLine(label, xs, ys, n, spec);
}

static void draw_line_lod(
    const char* label,
    const float* xs,
    const float* ys,
    int n,
    const ImVec4& col,
    bool dots,
    float weight = 1.5f
) {
    static thread_local Decimated buf; // reused to avoid reallocating
    const ImPlotRect lim = ImPlot::GetPlotLimits();
    const int px = std::max(1, (int)ImPlot::GetPlotSize().x);

    decimate_minmax(xs, ys, n, lim.X.Min, lim.X.Max, px, buf);
    const bool sparse = (int)buf.x.size() < px; // not decimated -> show dots
    draw_line(
        label,
        buf.x.data(),
        buf.y.data(),
        (int)buf.x.size(),
        col,
        dots && sparse,
        weight
    );
}

static void draw_crosses(
    const char* label,
    const float* xs,
    const float* ys,
    int n,
    const ImVec4& col
) {
    if (n <= 0)
        return;
    ImPlotSpec spec;
    spec.Marker = ImPlotMarker_Cross;
    spec.MarkerSize = 5.0f;
    spec.MarkerFillColor = col;
    spec.MarkerLineColor = col;
    ImPlot::PlotScatter(label, xs, ys, n, spec);
}

#include <cstdint>
#include <unordered_set>

// Draws crosses with LOD: skips points outside the view, and keeps at most one
// point per `cell_px` x `cell_px` screen cell. Pass fitting=true on the frame
// where SetNextAxesToFit is active so the auto-fit sees every point.
static void draw_crosses_lod(
    const char* label,
    const float* xs,
    const float* ys,
    int n,
    const ImVec4& col,
    bool fitting,
    float cell_px = 4.0f
) {
    if (n <= 0)
        return;

    // small sets or fit frame: draw everything
    if (fitting || n < 1000) {
        draw_crosses(label, xs, ys, n, col);
        return;
    }

    const ImPlotRect lim = ImPlot::GetPlotLimits();
    const ImVec2 sz = ImPlot::GetPlotSize();
    const float w = (float)(lim.X.Max - lim.X.Min),
                h = (float)(lim.Y.Max - lim.Y.Min);
    if (w <= 0.0 || h <= 0.0 || sz.x <= 0.0f || sz.y <= 0.0f) {
        draw_crosses(label, xs, ys, n, col);
        return;
    }
    const float sx = sz.x / w, sy = sz.y / h;

    static thread_local std::vector<float> bx, by;
    static thread_local std::unordered_set<uint64_t> seen;
    bx.clear();
    by.clear();
    seen.clear();

    const float margin = cell_px * 2; // keep crosses that straddle the edge
    for (int i = 0; i < n; ++i) {
        const float px = (float)(xs[i] - lim.X.Min) * sx;
        const float py = (float)(ys[i] - lim.Y.Min) * sy;
        if (px < -margin || px > sz.x + margin || py < -margin
            || py > sz.y + margin)
            continue;
        const uint64_t cx =
            (uint32_t)(int32_t)std::floor((px + margin) / cell_px);
        const uint64_t cy =
            (uint32_t)(int32_t)std::floor((py + margin) / cell_px);
        if (seen.insert((cx << 32) | cy).second) {
            bx.push_back(xs[i]);
            by.push_back(ys[i]);
        }
    }
    draw_crosses(label, bx.data(), by.data(), (int)bx.size(), col);
}

static ImVec4 body_col(int b) {
    static const ImVec4 pal[Body::MAX_BODIES] = {
        {0.30f, 0.62f, 0.95f, 1},
        {1.00f, 0.55f, 0.20f, 1},
        {0.30f, 0.85f, 0.50f, 1},
        {0.85f, 0.40f, 0.90f, 1},
        {0.95f, 0.85f, 0.30f, 1},
        {0.35f, 0.85f, 0.90f, 1},
        {0.95f, 0.45f, 0.45f, 1},
        {0.70f, 0.70f, 0.95f, 1}
    };
    return pal[b % Body::MAX_BODIES];
}

static ImVec4 dim(ImVec4 c) {
    c.x *= 0.6f;
    c.y *= 0.6f;
    c.z *= 0.6f;
    return c;
}

static ImVec4 faded(ImVec4 c) {
    c.w = 0.55f;
    return c;
}

static void draw_marker(
    const char* label,
    const float* x,
    const float* y,
    const ImVec4& col,
    bool filled
) {
    ImPlotSpec spec;
    spec.Marker = filled ? ImPlotMarker_Diamond : ImPlotMarker_Circle;
    spec.MarkerSize = 9.0f;
    spec.MarkerFillColor = filled ? col : ImVec4(0, 0, 0, 0);
    spec.MarkerLineColor = col;
    ImPlot::PlotScatter(label, x, y, 1, spec);
}

int main() {
    printf("-- Starting Filtration Station -- \n");
    AppState st;
    GenericViewer viewer(
        {"Filter viewer",
         1280,
         800,
         "filter_viewer.ini",
         {0.12f, 0.12f, 0.13f, 1.0f}}
    );

    // aliases so the rest of the code stays as it was
    Params& p = st.p;
    bool& show_xy = st.show_xy;
    bool& show_pos = st.show_pos;
    bool& show_vel = st.show_vel;
    bool& show_acc = st.show_acc;
    bool& show_err = st.show_err;
    bool& show_avg = st.show_avg;
    bool& auto_run = st.auto_run;
    bool (&show_body)[Body::MAX_BODIES] = st.show_body;
    std::vector<int>& order = st.order;

    p.seed_bodies();

    Result result;
    bool have_result = false;
    SimWorker worker; // all later runs happen on its thread
    bool refit = true;
    bool was_vis[PLOT_COUNT] =
        {}; // plots hidden last frame get fitted when shown again
    bool stale = false;

    return viewer.run(
        [&] {
            ImGuiSettingsHandler handler;
            handler.TypeName = "FilterViewer";
            handler.TypeHash = ImHashStr("FilterViewer");
            handler.UserData = &st;
            handler.ReadOpenFn = settings_open;
            handler.ReadLineFn = settings_line;
            handler.WriteAllFn = settings_write;
            ImGui::AddSettingsHandler(&handler);
        },
        [&] {
            sanitize(st);
            clamp_params(p);
            if (auto_run)
                worker.request(p);
        },
        [&] {
            ImGui::SetNextWindowPos({0, 0});
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin(
                "##root",
                nullptr,
                ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
                    | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
                    | ImGuiWindowFlags_NoBringToFrontOnFocus
            );

            // ---------------- controls (left) ----------------
            ImGui::BeginChild("controls", ImVec2(340, 0));
            bool changed = false;
            bool ui_dirty = false;

            const RunControls run_controls =
                viewer.drawRunControls(auto_run, worker.busy());
            const int hw = std::max(
                1,
                static_cast<int>(std::thread::hardware_concurrency())
            );
            ui_dirty |= ImGui::SliderInt("threads", &p.threads, 1, hw);
            ImGui::TextDisabled(
                "using %d of %d hardware threads",
                std::min(
                    p.threads,
                    p.signal == SignalType::SINE ? 1 : p.n_bodies
                ),
                hw
            );

            if (have_result) {
                const double ms = result.compute_ms;
                if (ms >= 1000.0)
                    ImGui::TextDisabled("last run: %.2f s", ms / 1000.0);
                else
                    ImGui::TextDisabled("last run: %.1f ms", ms);
                ImGui::TextDisabled(
                    "%d steps, %d bodies",
                    static_cast<int>(result.t.size()) - 1,
                    static_cast<int>(result.tracks.size())
                );
            }

            if (stale)
                ImGui::TextColored(
                    ImVec4(0.95f, 0.75f, 0.30f, 1.0f),
                    "parameters changed, press Run"
                );

            ImGui::TextDisabled("Ctrl+click any slider to type a value");

            ImGui::SeparatorText("Plots");
            ui_dirty |= ImGui::Checkbox("x-y path", &show_xy);
            ImGui::SameLine();
            ui_dirty |= ImGui::Checkbox("position", &show_pos);
            ui_dirty |= ImGui::Checkbox("velocity", &show_vel);
            ImGui::SameLine();
            ui_dirty |= ImGui::Checkbox("acceleration", &show_acc);
            ui_dirty |= ImGui::Checkbox("look-ahead error", &show_err);
            ImGui::SameLine();
            ui_dirty |= ImGui::Checkbox("avg accuracy", &show_avg);

            ImGui::SeparatorText("Filter type");
            changed |= ImGui::RadioButton(
                "Alpha-beta-gamma",
                (int*)(&p.filter),
                (int)FilterType::ABG
            );
            ImGui::SameLine();
            changed |= ImGui::RadioButton(
                "Kalman",
                (int*)(&p.filter),
                (int)FilterType::KALMAN
            );
            ImGui::SameLine();
            changed |= ImGui::RadioButton(
                "Polyfit",
                (int*)(&p.filter),
                (int)FilterType::POLYFIT
            );

            switch (p.filter) {
                case FilterType::ABG: {
                    ImGui::SeparatorText("Gains (both axes)");
                    changed |=
                        ImGui::SliderFloat("alpha", &p.alpha, 0.0f, 1.0f);
                    changed |= ImGui::SliderFloat("beta", &p.beta, 0.0f, 1.0f);
                    changed |=
                        ImGui::SliderFloat("gamma", &p.gamma, 0.0f, 1.0f);

                } break;
                case FilterType::KALMAN: {
                    ImGui::SeparatorText("Kalman tuning (both axes)");
                    changed |= ImGui::SliderFloat(
                        "process noise q",
                        &p.q,
                        0.001f,
                        100.0f,
                        "%.3f",
                        ImGuiSliderFlags_Logarithmic
                    );
                    changed |= ImGui::SliderFloat(
                        "meas. std (R)",
                        &p.meas_std,
                        1.0f,
                        20.0f,
                        "%.2f",
                        ImGuiSliderFlags_Logarithmic
                    );
                    changed |= ImGui::SliderFloat(
                        "initial variance",
                        &p.p0,
                        0.01f,
                        100.0f,
                        "%.2f",
                        ImGuiSliderFlags_Logarithmic
                    );
                } break;
                case FilterType::POLYFIT: {
                    ImGui::SeparatorText("Poly Fit yay");
                } break;
            }
            changed |= ImGui::SliderFloat("initial v", &p.v0, -2.0f, 2.0f);

            ImGui::SeparatorText("Sensor spec (1-sigma)");
            changed |=
                ImGui::Checkbox("use sensor noise for x/y/z", &p.use_sensor);
            changed |= ImGui::SliderFloat(
                "[x y z] sigma [m]",
                &p.sig_pos,
                0.0f,
                20.0f
            );
            changed |=
                ImGui::SliderFloat("r_c sigma [m]", &p.sig_rc, 0.0f, 20.0f);
            changed |= ImGui::SliderFloat(
                "v_r sigma [m/s]",
                &p.sig_vr,
                0.0f,
                1.0f,
                "%.3f"
            );
            changed |= ImGui::SliderFloat(
                "diameter sigma [frac]",
                &p.sig_diam_frac,
                0.0f,
                0.5f,
                "%.3f"
            );

            ImGui::SeparatorText("Run length");
            changed |=
                ImGui::SliderFloat("duration [s]", &p.duration, 0.5f, 60.0f);
            changed |= ImGui::SliderFloat("dt [s]", &p.dt, 0.01f, 0.5f);
            ImGui::Text(
                "%d steps",
                std::max(1, static_cast<int>(std::lround(p.duration / p.dt)))
            );

            ImGui::SeparatorText("Prediction");
            changed |=
                ImGui::SliderFloat("look-ahead [s]", &p.horizon, 0.0f, 5.0f);
            changed |= ImGui::SliderFloat(
                "sweep max [s]",
                &p.max_horizon,
                0.1f,
                10.0f
            );
            changed |=
                ImGui::SliderFloat("warm-up [s]", &p.warmup, 0.0f, 10.0f);

            ImGui::SeparatorText("Signal");
            changed |= ImGui::RadioButton(
                "Sinusoid",
                (int*)&p.signal,
                (int)SignalType::SINE
            );
            ImGui::SameLine();
            changed |= ImGui::RadioButton(
                "Const-accel bodies",
                (int*)&p.signal,
                (int)SignalType::BODIES
            );

            ImGui::SeparatorText("Measurement");
            if (p.signal == SignalType::SINE) {
                ImGui::TextDisabled("actual = A * sin(omega * t + B)");
                changed |= ImGui::SliderFloat("A x", &p.amp_x, 0.0f, 5.0f);
                changed |= ImGui::SliderFloat(
                    "omega x [rad/s]",
                    &p.omega_x,
                    0.1f,
                    5.0f
                );
                changed |= ImGui::SliderFloat(
                    "B x [rad]",
                    &p.phase_x,
                    -3.1416f,
                    3.1416f,
                    "%.2f"
                );
                changed |= ImGui::SliderFloat("A y", &p.amp_y, 0.0f, 5.0f);
                changed |= ImGui::SliderFloat(
                    "omega y [rad/s]",
                    &p.omega_y,
                    0.1f,
                    5.0f
                );
                changed |= ImGui::SliderFloat(
                    "B y [rad]",
                    &p.phase_y,
                    -3.1416f,
                    3.1416f,
                    "%.2f"
                );
            }
            changed |= ImGui::SliderFloat("noise std", &p.noise, 0.0f, 1.0f);
            int changed_seed = p.seed;
            bool new_seed = false;
            ImGui::InputScalar("Seed", ImGuiDataType_S32, &changed_seed);
            if (ImGui::Button("New seed")) {
                ++changed_seed;
            }
            if (changed_seed != p.seed) {
                new_seed = true;
                changed = true;
            }

            if (p.signal == SignalType::BODIES) {
                changed |= ImGui::SliderInt(
                    "bodies",
                    &p.n_bodies,
                    1,
                    Body::MAX_BODIES
                );
                for (int b = 0; b < p.n_bodies; ++b) {
                    ImGui::PushID(b);
                    ImGui::PushStyleColor(ImGuiCol_Text, body_col(b));
                    const bool open = ImGui::TreeNode("body", "body %d", b);
                    ImGui::PopStyleColor();
                    if (open) {
                        Body& B = p.bodies[b];
                        ImGui::Checkbox("show", &show_body[b]);
                        changed |= ImGui::SliderFloat3(
                            "start pos",
                            &B.x0,
                            -5.0f,
                            5.0f
                        );
                        changed |= ImGui::SliderFloat3(
                            "start vel",
                            &B.vx,
                            -3.0f,
                            3.0f
                        );
                        changed |=
                            ImGui::SliderFloat3("accel", &B.ax, -2.0f, 2.0f);
                        changed |= ImGui::SliderFloat(
                            "diameter [m]",
                            &B.diameter,
                            1.0f,
                            20000.0f
                        );
                        ImGui::TreePop();
                    }
                    ImGui::PopID();
                }
            }
            if (new_seed) {
                p.seed = changed_seed;
                p.seed_bodies();
            }

            ImGui::EndChild();

            // Typed values may go outside a slider's range, so keep the ones that would break
            // the maths (division by zero, negative std, runaway step counts) in a safe range.
            clamp_params(p);

            // Re-run the calculation whenever anything changed.
            if (changed) {
                ImGui::MarkIniSettingsDirty();
                if (!auto_run)
                    stale = true;
            }
            if (ui_dirty || run_controls.auto_run_changed)
                ImGui::MarkIniSettingsDirty();

            if (run_controls.run_requested
                || (auto_run && (changed || stale))) {
                worker.request(p);
                stale = false;
            }

            // Swap in a finished result (computed on the worker thread) when one is ready.
            // Only fit the axes for the very first result; later runs keep the user's view.
            // (Double-click a plot to fit it manually.)
            if (auto fresh = worker.take()) {
                result = std::move(*fresh);
                if (!have_result)
                    refit = true;
                have_result = true;
            }

            // ---------------- plots (right) ----------------
            ImGui::SameLine();
            ImGui::BeginChild("plots", ImVec2(0, 0));

            if (!have_result) {
                if (worker.busy())
                    ImGui::TextDisabled("computing...");
                else
                    ImGui::TextDisabled("No results yet. Press Run.");
            } else {
                const bool vis[PLOT_COUNT] =
                    {show_xy, show_pos, show_vel, show_acc, show_err, show_avg};

                const Track& k0 = result.tracks.front();
                const int n = static_cast<int>(
                    std::min(k0.fx->x_k.size(), k0.fy->x_k.size())
                );
                const int m = static_cast<int>(k0.zx.size());
                const float* t = result.t.data();

                auto draw_plot = [&](int id) {
                    const bool fit = refit || !was_vis[id];
                    was_vis[id] = true;
                    if (fit)
                        ImPlot::SetNextAxesToFit();
                    const ImPlotFlags base = ImPlotFlags_NoTitle;
                    const int nt = static_cast<int>(result.tracks.size());
                    char lab[48];
                    auto L = [&](const char* s, int b) {
                        std::snprintf(lab, sizeof lab, "b%d %s", b, s);
                        return static_cast<const char*>(lab);
                    };

                    switch (id) {
                        case PLOT_XY:
                            if (ImPlot::BeginPlot(
                                    "##xy",
                                    ImVec2(-1, -1),
                                    base | ImPlotFlags_Equal
                                )) {
                                ImPlot::SetupAxes("x", "y");
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    const ImVec4 c = body_col(b);
                                    draw_line_lod(
                                        L("true", b),
                                        k.tx_ext.data(),
                                        k.ty_ext.data(),
                                        (int)k.tx_ext.size(),
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_crosses_lod(
                                        L("meas", b),
                                        k.zx.data(),
                                        k.zy.data(),
                                        m,
                                        kColMeas,
                                        fit
                                    );
                                    draw_line_lod(
                                        L("estimate", b),
                                        k.fx->x_k.data(),
                                        k.fy->x_k.data(),
                                        n,
                                        c,
                                        true
                                    );
                                    draw_line_lod(
                                        L("forecast", b),
                                        k.px.data(),
                                        k.py.data(),
                                        (int)k.px.size(),
                                        faded(c),
                                        false,
                                        1.0f
                                    );
                                    draw_marker(
                                        L("predicted end", b),
                                        &k.fut_x,
                                        &k.fut_y,
                                        c,
                                        true
                                    );
                                    draw_marker(
                                        L("true end", b),
                                        &k.fut_tx,
                                        &k.fut_ty,
                                        c,
                                        false
                                    );
                                }
                                ImPlot::EndPlot();
                            }
                            break;
                        case PLOT_POS:
                            if (ImPlot::BeginPlot(
                                    "##pos",
                                    ImVec2(-1, -1),
                                    base
                                )) {
                                ImPlot::SetupAxes("time [s]", "position");
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    const ImVec4 c = body_col(b);
                                    draw_line_lod(
                                        L("x true", b),
                                        t + 1,
                                        k.tx.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("y true", b),
                                        t + 1,
                                        k.ty.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("x", b),
                                        t,
                                        k.fx->x_k.data(),
                                        n,
                                        c,
                                        true
                                    );
                                    draw_line_lod(
                                        L("y", b),
                                        t,
                                        k.fy->x_k.data(),
                                        n,
                                        dim(c),
                                        true
                                    );
                                    draw_line_lod(
                                        L("x forecast", b),
                                        k.pt.data(),
                                        k.px.data(),
                                        (int)k.px.size(),
                                        faded(c),
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("y forecast", b),
                                        k.pt.data(),
                                        k.py.data(),
                                        (int)k.py.size(),
                                        faded(dim(c)),
                                        false,
                                        1.0f
                                    );
                                }
                                ImPlot::EndPlot();
                            }
                            break;
                        case PLOT_VEL:
                            if (ImPlot::BeginPlot(
                                    "##vel",
                                    ImVec2(-1, -1),
                                    base
                                )) {
                                ImPlot::SetupAxes("time [s]", "velocity");
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    const ImVec4 c = body_col(b);
                                    draw_line_lod(
                                        L("x true", b),
                                        t + 1,
                                        k.tvx.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("y true", b),
                                        t + 1,
                                        k.tvy.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("x", b),
                                        t,
                                        k.fx->v_k.data(),
                                        n,
                                        c,
                                        true
                                    );
                                    draw_line_lod(
                                        L("y", b),
                                        t,
                                        k.fy->v_k.data(),
                                        n,
                                        dim(c),
                                        true
                                    );
                                }
                                ImPlot::EndPlot();
                            }
                            break;
                        case PLOT_ACC:
                            if (ImPlot::BeginPlot(
                                    "##acc",
                                    ImVec2(-1, -1),
                                    base
                                )) {
                                ImPlot::SetupAxes("time [s]", "acceleration");
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    const ImVec4 c = body_col(b);
                                    draw_line_lod(
                                        L("x true", b),
                                        t + 1,
                                        k.tax.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("y true", b),
                                        t + 1,
                                        k.tay.data(),
                                        m,
                                        kColActual,
                                        false,
                                        1.0f
                                    );
                                    draw_line_lod(
                                        L("x", b),
                                        t,
                                        k.fx->a_k.data(),
                                        n,
                                        c,
                                        true
                                    );
                                    draw_line_lod(
                                        L("y", b),
                                        t,
                                        k.fy->a_k.data(),
                                        n,
                                        dim(c),
                                        true
                                    );
                                }
                                ImPlot::EndPlot();
                            }
                            break;
                        case PLOT_ERR:
                            if (ImPlot::BeginPlot(
                                    "##err",
                                    ImVec2(-1, -1),
                                    base
                                )) {
                                ImPlot::SetupAxes("look-ahead [s]", "error");
                                const int ns =
                                    static_cast<int>(result.sweep_h.size());
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    const ImVec4 c = body_col(b);
                                    draw_line_lod(
                                        L("x RMS", b),
                                        result.sweep_h.data(),
                                        k.rms_x.data(),
                                        ns,
                                        c,
                                        false
                                    );
                                    draw_line_lod(
                                        L("y RMS", b),
                                        result.sweep_h.data(),
                                        k.rms_y.data(),
                                        ns,
                                        dim(c),
                                        false
                                    );
                                    if (!k.sig_x.empty())
                                        draw_line_lod(
                                            L("x sigma", b),
                                            result.sweep_h.data(),
                                            k.sig_x.data(),
                                            ns,
                                            faded(c),
                                            false,
                                            1.0f
                                        );
                                }
                                double hz = result.params.horizon;
                                ImPlotSpec spec;
                                spec.LineColor = kColFcast;
                                ImPlot::PlotInfLines(
                                    "look-ahead",
                                    &hz,
                                    1,
                                    spec
                                );
                                ImPlot::EndPlot();
                            }
                            break;
                        case PLOT_AVG:
                            if (ImPlot::BeginPlot(
                                    "##avg",
                                    ImVec2(-1, -1),
                                    base
                                )) {
                                ImPlot::SetupAxes(
                                    "time [s]",
                                    "position error at look-ahead"
                                );
                                for (int b = 0; b < nt; ++b) {
                                    if (!show_body[b])
                                        continue;
                                    const Track& k = result.tracks[b];
                                    draw_line_lod(
                                        L("err", b),
                                        t + 1,
                                        k.err.data(),
                                        m,
                                        faded(body_col(b)),
                                        false,
                                        1.0f
                                    );
                                }
                                draw_line_lod(
                                    "mean of all bodies",
                                    t + 1,
                                    result.mean_err.data(),
                                    m,
                                    ImVec4(0.95f, 0.95f, 0.95f, 1.0f),
                                    false,
                                    2.5f
                                );
                                char avg_lab[64];
                                std::snprintf(
                                    avg_lab,
                                    sizeof avg_lab,
                                    "running avg (%.3f)##runavg",
                                    result.mean_err_scored
                                );
                                draw_line_lod(
                                    avg_lab,
                                    t + 1,
                                    result.run_avg.data(),
                                    m,
                                    kColFcast,
                                    false,
                                    2.0f
                                );
                                ImPlot::EndPlot();
                            }
                            break;
                    }
                };
                for (int id = 0; id < PLOT_COUNT; ++id)
                    if (!vis[id])
                        was_vis[id] = false;
                std::vector<int> vis_ids;
                for (int id : order)
                    if (vis[id])
                        vis_ids.push_back(id);
                const int cnt = static_cast<int>(vis_ids.size());

                if (cnt == 0) {
                    ImGui::TextDisabled("No plots selected.");
                } else {
                    const int cols = cnt > 1 ? 2 : 1;
                    const int rows = (cnt + cols - 1) / cols;
                    const ImVec2 avail = ImGui::GetContentRegionAvail();
                    const ImVec2 sp = ImGui::GetStyle().ItemSpacing;
                    const float cw =
                        (avail.x - sp.x * (float)(cols - 1)) / (float)cols;
                    const float ch =
                        (avail.y - sp.y * (float)(rows - 1)) / (float)rows;
                    const ImVec2 origin = ImGui::GetCursorPos();

                    int swap_a = -1, swap_b = -1;
                    for (int k = 0; k < cnt; ++k) {
                        const int id = vis_ids[k];
                        const int r = k / cols, c = k % cols;
                        const bool spans =
                            cols == 2 && k == cnt - 1 && c == 0; // odd one out
                        ImGui::SetCursorPos(
                            {origin.x + (float)c * (cw + sp.x),
                             origin.y + (float)r * (ch + sp.y)}
                        );

                        ImGui::PushID(id);
                        ImGui::BeginChild(
                            "cell",
                            ImVec2(spans ? avail.x : cw, ch),
                            ImGuiChildFlags_None,
                            ImGuiWindowFlags_NoScrollbar
                                | ImGuiWindowFlags_NoScrollWithMouse
                        );

                        // drag handle: grab to move, drop on another cell's handle to swap
                        char label[64];
                        std::snprintf(
                            label,
                            sizeof label,
                            "::  %s##hdl",
                            kPlotTitle[id]
                        );
                        ImGui::Selectable(label, false);
                        if (ImGui::BeginDragDropSource()) {
                            ImGui::SetDragDropPayload("PLOT", &id, sizeof(int));
                            ImGui::TextUnformatted(kPlotTitle[id]);
                            ImGui::EndDragDropSource();
                        }
                        if (ImGui::BeginDragDropTarget()) {
                            if (const ImGuiPayload* pl =
                                    ImGui::AcceptDragDropPayload("PLOT")) {
                                swap_a = *static_cast<const int*>(pl->Data);
                                swap_b = id;
                            }
                            ImGui::EndDragDropTarget();
                        }

                        draw_plot(id);

                        ImGui::EndChild();
                        ImGui::PopID();
                    }

                    if (swap_a >= 0 && swap_a != swap_b) {
                        auto ia = std::find(order.begin(), order.end(), swap_a);
                        auto ib = std::find(order.begin(), order.end(), swap_b);
                        std::iter_swap(ia, ib);
                    }
                }
            }
            ImGui::EndChild();
            refit = false;
            ImGui::End();
        }
    );
}
