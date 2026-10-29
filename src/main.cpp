// 2D tracking filter viewer: alpha-beta-gamma or Kalman (one independent filter per axis).
// Everything is drawn in a single Dear ImGui window: controls on the left, ImPlot plots on
// the right. Whenever a control changes, on auto the filter is re-run from scratch over a fixed time
// span (the "duration" slider); the plots just redraw every frame.

#include "filters.hpp"
#include "plotting.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_internal.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

// Runs simulations on a background thread so the UI never waits on the maths.
// Requests are coalesced: if several arrive while a run is in progress, only the most
// recent one is computed next. Finished results are picked up by the UI with take().
class SimWorker {
  public:
    SimWorker() : thread_([this] { loop(); }) {}

    ~SimWorker() {
        {
            std::lock_guard<std::mutex> lk(m_);
            quit_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }

    SimWorker(const SimWorker&) = delete;
    SimWorker& operator=(const SimWorker&) = delete;

    // Ask for a run with these parameters (replaces any request not yet started).
    void request(const Params& p) {
        {
            std::lock_guard<std::mutex> lk(m_);
            pending_ = p;
        }
        cv_.notify_one();
    }

    // Returns the newest finished result, if there is one the UI hasn't taken yet.
    std::optional<Result> take() {
        std::optional<Result> r;
        std::lock_guard<std::mutex> lk(m_);
        r.swap(ready_);
        return r;
    }

    bool busy() {
        std::lock_guard<std::mutex> lk(m_);
        return running_ || pending_.has_value();
    }

  private:
    void loop() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [this] { return quit_ || pending_.has_value(); });
            if (quit_)
                return;

            const Params p = *pending_;
            pending_.reset();
            running_ = true;

            lk.unlock();
            Result r = run_sim(p); // the heavy part, with the mutex released
            lk.lock();

            running_ = false;
            ready_ = std::move(r);
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::optional<Params> pending_;
    std::optional<Result> ready_;
    bool running_ = false;
    bool quit_ = false;
    std::thread
        thread_; // last, so everything above exists before the thread starts
};

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

static void draw_crosses(
    const char* label,
    const double* xs,
    const double* ys,
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
    const double* x,
    const double* y,
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
    if (!glfwInit())
        return 1;
    GLFWwindow* window =
        glfwCreateWindow(1280, 800, "Filter viewer", nullptr, nullptr);
    if (!window)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    AppState st;
    ImGuiSettingsHandler handler;
    handler.TypeName = "FilterViewer";
    handler.TypeHash = ImHashStr("FilterViewer");
    handler.UserData = &st;
    handler.ReadOpenFn = settings_open;
    handler.ReadLineFn = settings_line;
    handler.WriteAllFn = settings_write;
    ImGui::AddSettingsHandler(&handler);

    ImGui::GetIO().IniFilename = "filter_viewer.ini"; // optional: its own file
    ImGui::LoadIniSettingsFromDisk(ImGui::GetIO().IniFilename);
    sanitize(st);

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

    clamp_params(p);
    Result result;
    bool have_result = false;
    SimWorker worker; // all later runs happen on its thread
    bool refit = true;
    bool stale = false;

    if (auto_run)
        worker.request(p);

    if (auto fresh = worker.take()) {
        result = std::move(*fresh);
        have_result = true;
        refit = true;
    }

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

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
        bool run_now = false;
        bool mode_changed = false;
        bool ui_dirty = false;

        ImGui::SeparatorText("Run");
        int mode = auto_run ? 0 : 1;
        mode_changed |= ImGui::RadioButton("Auto rerun", &mode, 0);
        ImGui::SameLine();
        mode_changed |= ImGui::RadioButton("Manual", &mode, 1);
        auto_run = mode == 0;
        const int hw =
            std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        ui_dirty |= ImGui::SliderInt("threads", &p.threads, 1, hw);
        ImGui::TextDisabled(
            "using %d of %d hardware threads",
            std::min(p.threads, p.signal == SignalType::SINE ? 1 : p.n_bodies),
            hw
        );

        if (ImGui::Button("Run", ImVec2(-1, 0))) {
            run_now = true;
        }

        ImGui::TextDisabled(worker.busy() ? "computing..." : "chilling...");
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

        if (p.filter == FilterType::ABG) {
            ImGui::SeparatorText("Gains (both axes)");
            changed |= ImGui::SliderFloat("alpha", &p.alpha, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("beta", &p.beta, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("gamma", &p.gamma, 0.0f, 1.0f);
        } else {
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
                0.001f,
                1.0f,
                "%.3f",
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
        }
        changed |= ImGui::SliderFloat("initial v", &p.v0, -2.0f, 2.0f);

        ImGui::SeparatorText("Run length");
        changed |= ImGui::SliderFloat("duration [s]", &p.duration, 0.5f, 60.0f);
        changed |= ImGui::SliderFloat("dt [s]", &p.dt, 0.01f, 0.5f);
        ImGui::Text(
            "%d steps",
            std::max(1, static_cast<int>(std::lround(p.duration / p.dt)))
        );

        ImGui::SeparatorText("Prediction");
        changed |= ImGui::SliderFloat("look-ahead [s]", &p.horizon, 0.0f, 5.0f);
        changed |=
            ImGui::SliderFloat("sweep max [s]", &p.max_horizon, 0.1f, 10.0f);
        changed |= ImGui::SliderFloat("warm-up [s]", &p.warmup, 0.0f, 10.0f);

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
            changed |=
                ImGui::SliderFloat("omega x [rad/s]", &p.omega_x, 0.1f, 5.0f);
            changed |= ImGui::SliderFloat(
                "B x [rad]",
                &p.phase_x,
                -3.1416f,
                3.1416f,
                "%.2f"
            );
            changed |= ImGui::SliderFloat("A y", &p.amp_y, 0.0f, 5.0f);
            changed |=
                ImGui::SliderFloat("omega y [rad/s]", &p.omega_y, 0.1f, 5.0f);
            changed |= ImGui::SliderFloat(
                "B y [rad]",
                &p.phase_y,
                -3.1416f,
                3.1416f,
                "%.2f"
            );
        }
        changed |= ImGui::SliderFloat("noise std", &p.noise, 0.0f, 1.0f);
        bool new_seed = false;
        if (ImGui::Button("New noise seed")) {
            new_seed = true;
            changed = true;
        }

        if (p.signal == SignalType::BODIES) {
            changed |=
                ImGui::SliderInt("bodies", &p.n_bodies, 1, Body::MAX_BODIES);
            for (int b = 0; b < p.n_bodies; ++b) {
                ImGui::PushID(b);
                ImGui::PushStyleColor(ImGuiCol_Text, body_col(b));
                const bool open = ImGui::TreeNode("body", "body %d", b);
                ImGui::PopStyleColor();
                if (open) {
                    Body& B = p.bodies[b];
                    ImGui::Checkbox("show", &show_body[b]);
                    changed |=
                        ImGui::SliderFloat2("start pos", &B.x0, -5.0f, 5.0f);
                    changed |=
                        ImGui::SliderFloat2("start vel", &B.vx, -3.0f, 3.0f);
                    changed |= ImGui::SliderFloat2("accel", &B.ax, -2.0f, 2.0f);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
        if (new_seed)
            ++p.seed;

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
        if (mode_changed || ui_dirty)
            ImGui::MarkIniSettingsDirty();

        if (run_now || (auto_run && (changed || stale))) {
            worker.request(p);
            stale = false;
        }

        // Swap in a finished result (computed on the worker thread) when one is ready.
        if (auto fresh = worker.take()) {
            result = std::move(*fresh);
            have_result = true;
            refit = true;
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
            const double* t = result.t.data();

            auto draw_plot = [&](int id) {
                if (refit)
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
                                draw_line(
                                    L("true", b),
                                    k.tx.data(),
                                    k.ty.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_crosses(
                                    L("meas", b),
                                    k.zx.data(),
                                    k.zy.data(),
                                    m,
                                    kColMeas
                                );
                                draw_line(
                                    L("estimate", b),
                                    k.fx->x_k.data(),
                                    k.fy->x_k.data(),
                                    n,
                                    c,
                                    true
                                );
                                draw_line(
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
                        if (ImPlot::BeginPlot("##pos", ImVec2(-1, -1), base)) {
                            ImPlot::SetupAxes("time [s]", "position");
                            for (int b = 0; b < nt; ++b) {
                                if (!show_body[b])
                                    continue;
                                const Track& k = result.tracks[b];
                                const ImVec4 c = body_col(b);
                                draw_line(
                                    L("x true", b),
                                    t + 1,
                                    k.tx.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("y true", b),
                                    t + 1,
                                    k.ty.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("x", b),
                                    t,
                                    k.fx->x_k.data(),
                                    n,
                                    c,
                                    true
                                );
                                draw_line(
                                    L("y", b),
                                    t,
                                    k.fy->x_k.data(),
                                    n,
                                    dim(c),
                                    true
                                );
                                draw_line(
                                    L("x forecast", b),
                                    k.pt.data(),
                                    k.px.data(),
                                    (int)k.px.size(),
                                    faded(c),
                                    false,
                                    1.0f
                                );
                                draw_line(
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
                        if (ImPlot::BeginPlot("##vel", ImVec2(-1, -1), base)) {
                            ImPlot::SetupAxes("time [s]", "velocity");
                            for (int b = 0; b < nt; ++b) {
                                if (!show_body[b])
                                    continue;
                                const Track& k = result.tracks[b];
                                const ImVec4 c = body_col(b);
                                draw_line(
                                    L("x true", b),
                                    t + 1,
                                    k.tvx.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("y true", b),
                                    t + 1,
                                    k.tvy.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("x", b),
                                    t,
                                    k.fx->v_k.data(),
                                    n,
                                    c,
                                    true
                                );
                                draw_line(
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
                        if (ImPlot::BeginPlot("##acc", ImVec2(-1, -1), base)) {
                            ImPlot::SetupAxes("time [s]", "acceleration");
                            for (int b = 0; b < nt; ++b) {
                                if (!show_body[b])
                                    continue;
                                const Track& k = result.tracks[b];
                                const ImVec4 c = body_col(b);
                                draw_line(
                                    L("x true", b),
                                    t + 1,
                                    k.tax.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("y true", b),
                                    t + 1,
                                    k.tay.data(),
                                    m,
                                    kColActual,
                                    false,
                                    1.0f
                                );
                                draw_line(
                                    L("x", b),
                                    t,
                                    k.fx->a_k.data(),
                                    n,
                                    c,
                                    true
                                );
                                draw_line(
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
                        if (ImPlot::BeginPlot("##err", ImVec2(-1, -1), base)) {
                            ImPlot::SetupAxes("look-ahead [s]", "error");
                            const int ns =
                                static_cast<int>(result.sweep_h.size());
                            for (int b = 0; b < nt; ++b) {
                                if (!show_body[b])
                                    continue;
                                const Track& k = result.tracks[b];
                                const ImVec4 c = body_col(b);
                                draw_line(
                                    L("x RMS", b),
                                    result.sweep_h.data(),
                                    k.rms_x.data(),
                                    ns,
                                    c,
                                    false
                                );
                                draw_line(
                                    L("y RMS", b),
                                    result.sweep_h.data(),
                                    k.rms_y.data(),
                                    ns,
                                    dim(c),
                                    false
                                );
                                if (!k.sig_x.empty())
                                    draw_line(
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
                            ImPlot::PlotInfLines("look-ahead", &hz, 1, spec);
                            ImPlot::EndPlot();
                        }
                        break;
                    case PLOT_AVG:
                        if (ImPlot::BeginPlot("##avg", ImVec2(-1, -1), base)) {
                            ImPlot::SetupAxes(
                                "time [s]",
                                "position error at look-ahead"
                            );
                            for (int b = 0; b < nt; ++b) {
                                if (!show_body[b])
                                    continue;
                                const Track& k = result.tracks[b];
                                draw_line(
                                    L("err", b),
                                    t + 1,
                                    k.err.data(),
                                    m,
                                    faded(body_col(b)),
                                    false,
                                    1.0f
                                );
                            }
                            draw_line(
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
                            draw_line(
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
                const float cw = (avail.x - sp.x * (cols - 1)) / cols;
                const float ch = (avail.y - sp.y * (rows - 1)) / rows;
                const ImVec2 origin = ImGui::GetCursorPos();

                int swap_a = -1, swap_b = -1;
                for (int k = 0; k < cnt; ++k) {
                    const int id = vis_ids[k];
                    const int r = k / cols, c = k % cols;
                    const bool spans =
                        cols == 2 && k == cnt - 1 && c == 0; // odd one out
                    ImGui::SetCursorPos(
                        {origin.x + c * (cw + sp.x), origin.y + r * (ch + sp.y)}
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

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.12f, 0.12f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
