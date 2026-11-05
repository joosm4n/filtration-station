#include "physics.hpp"
#include "simulation_worker.hpp"
#include "viewer_base.hpp"

#include <imgui_internal.h>
#include <implot.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

namespace {

using PhysicsWorker =
    SimulationWorker<PhysicsParams, PhysicsResult, run_sim>;

struct PhysicsAppState {
    PhysicsParams params;
    bool auto_run = true;
};

template<class F>
void visit_settings(PhysicsAppState& state, F&& visit) {
    visit("n_bodies", state.params.n_bodies);
    visit("dt", state.params.dt);
    visit("duration", state.params.duration);
    visit("gravity_x", state.params.gravity_x);
    visit("gravity_y", state.params.gravity_y);
    visit("restitution", state.params.restitution);
    visit("damping", state.params.damping);
    visit("min_x", state.params.min_x);
    visit("max_x", state.params.max_x);
    visit("min_y", state.params.min_y);
    visit("max_y", state.params.max_y);
    visit("threads", state.params.threads);
    visit("auto_run", state.auto_run);
}

void put_setting(ImGuiTextBuffer* output, const char* key, float value) {
    output->appendf("%s=%.9g\n", key, value);
}

void put_setting(ImGuiTextBuffer* output, const char* key, int value) {
    output->appendf("%s=%d\n", key, value);
}

void put_setting(ImGuiTextBuffer* output, const char* key, bool value) {
    output->appendf("%s=%d\n", key, value ? 1 : 0);
}

void get_setting(const char* value, float& target) {
    target = std::strtof(value, nullptr);
}

void get_setting(const char* value, int& target) {
    target = std::atoi(value);
}

void get_setting(const char* value, bool& target) {
    target = std::atoi(value) != 0;
}

void* settings_open(
    ImGuiContext*,
    ImGuiSettingsHandler* handler,
    const char* name
) {
    return std::strcmp(name, "State") == 0 ? handler->UserData : nullptr;
}

void settings_line(
    ImGuiContext*,
    ImGuiSettingsHandler*,
    void* entry,
    const char* line
) {
    auto& state = *static_cast<PhysicsAppState*>(entry);
    const char* equals = std::strchr(line, '=');
    if (!equals)
        return;
    const std::string key(line, equals - line);
    const char* value = equals + 1;
    if (key.rfind("body", 0) == 0 && key.size() > 4
        && std::isdigit(static_cast<unsigned char>(key[4]))) {
        const int index = std::atoi(key.c_str() + 4);
        if (index >= 0 && index < PhysicsParams::MAX_BODIES)
            std::sscanf(
                value,
                "%f,%f,%f,%f,%f,%f",
                &state.params.bodies[index].x,
                &state.params.bodies[index].y,
                &state.params.bodies[index].vx,
                &state.params.bodies[index].vy,
                &state.params.bodies[index].mass,
                &state.params.bodies[index].radius
            );
        return;
    }
    visit_settings(state, [&](const char* setting, auto& target) {
        if (key == setting)
            get_setting(value, target);
    });
}

void settings_write(
    ImGuiContext*,
    ImGuiSettingsHandler* handler,
    ImGuiTextBuffer* output
) {
    auto& state = *static_cast<PhysicsAppState*>(handler->UserData);
    output->appendf("[%s][State]\n", handler->TypeName);
    visit_settings(state, [&](const char* key, auto& value) {
        put_setting(output, key, value);
    });
    for (int i = 0; i < PhysicsParams::MAX_BODIES; ++i) {
        const auto& body = state.params.bodies[i];
        output->appendf(
            "body%d=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
            i,
            body.x,
            body.y,
            body.vx,
            body.vy,
            body.mass,
            body.radius
        );
    }
    output->append("\n");
}

ImVec4 body_color(int index) {
    static const ImVec4 colors[] = {
        {0.30f, 0.62f, 0.95f, 1.0f},
        {1.00f, 0.55f, 0.20f, 1.0f},
        {0.30f, 0.85f, 0.50f, 1.0f},
        {0.85f, 0.40f, 0.90f, 1.0f},
        {0.95f, 0.85f, 0.30f, 1.0f},
        {0.35f, 0.85f, 0.90f, 1.0f},
    };
    return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

} // namespace

int main() {
    PhysicsAppState state;
    GenericViewer viewer(
        {"2D physics viewer", 1100, 750, "physics_viewer.ini", {0.08f, 0.08f, 0.10f, 1.0f}}
    );

    PhysicsParams& params = state.params;
    PhysicsResult result;
    PhysicsWorker worker;
    bool have_result = false;
    bool& auto_run = state.auto_run;
    worker.request(params);

    return viewer.run(
        [&] {
            ImGuiSettingsHandler settings;
            settings.TypeName = "PhysicsViewer";
            settings.TypeHash = ImHashStr("PhysicsViewer");
            settings.UserData = &state;
            settings.ReadOpenFn = settings_open;
            settings.ReadLineFn = settings_line;
            settings.WriteAllFn = settings_write;
            ImGui::AddSettingsHandler(&settings);
        },
        [&] { clamp_params(state.params); },
        [&] {
        if (auto ready = worker.take()) {
            result = std::move(*ready);
            have_result = true;
        }

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin(
            "##physics-root",
            nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
        );
        ImGui::BeginChild("physics-controls", {300, 0}, true);
        bool changed = false;
        const RunControls run_controls =
            viewer.drawRunControls(auto_run, worker.busy());
        const int hardware_threads = std::max(
            1,
            static_cast<int>(std::thread::hardware_concurrency())
        );
        ImGui::SeparatorText("Simulation");
        changed |= ImGui::SliderInt(
            "bodies",
            &params.n_bodies,
            1,
            PhysicsParams::MAX_BODIES
        );
        changed |= ImGui::SliderFloat("duration [s]", &params.duration, 1.0f, 60.0f);
        changed |= ImGui::SliderFloat("dt [s]", &params.dt, 0.001f, 0.05f);
        changed |= ImGui::SliderInt(
            "threads",
            &params.threads,
            1,
            std::min(hardware_threads, PhysicsParams::MAX_BODIES)
        );
        ImGui::TextDisabled(
            "using %d of %d hardware threads",
            std::min(params.threads, params.n_bodies),
            hardware_threads
        );
        changed |= ImGui::SliderFloat("gravity x", &params.gravity_x, -20.0f, 20.0f);
        changed |= ImGui::SliderFloat("gravity y", &params.gravity_y, -20.0f, 20.0f);
        changed |= ImGui::SliderFloat(
            "restitution",
            &params.restitution,
            0.0f,
            1.0f
        );
        changed |= ImGui::SliderFloat("damping", &params.damping, 0.9f, 1.0f);
        ImGui::SeparatorText("Bounds");
        changed |= ImGui::SliderFloat("min x", &params.min_x, -20.0f, 0.0f);
        changed |= ImGui::SliderFloat("max x", &params.max_x, 0.0f, 20.0f);
        changed |= ImGui::SliderFloat("min y", &params.min_y, -20.0f, 0.0f);
        changed |= ImGui::SliderFloat("max y", &params.max_y, 0.0f, 20.0f);
        if (ImGui::TreeNode("Bodies")) {
            for (int i = 0; i < params.n_bodies; ++i) {
                ImGui::PushID(i);
                if (ImGui::TreeNode("body", "body %d", i)) {
                    PhysicsBody& body = params.bodies[i];
                    changed |= ImGui::SliderFloat2(
                        "start position",
                        &body.x,
                        -20.0f,
                        20.0f
                    );
                    changed |= ImGui::SliderFloat2(
                        "start velocity",
                        &body.vx,
                        -20.0f,
                        20.0f
                    );
                    changed |=
                        ImGui::SliderFloat("mass", &body.mass, 0.01f, 20.0f);
                    changed |= ImGui::SliderFloat(
                        "radius",
                        &body.radius,
                        0.01f,
                        2.0f
                    );
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        clamp_params(params);
        if (run_controls.run_requested
            || (auto_run && changed))
            worker.request(params);
        if (changed || run_controls.auto_run_changed)
            ImGui::MarkIniSettingsDirty();
        if (have_result)
            ImGui::TextDisabled(
                "%d steps, %.1f ms",
                static_cast<int>(result.t.size()) - 1,
                result.compute_ms
            );
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("physics-plots", {0, 0}, true);
        if (have_result && ImPlot::BeginPlot("2D positions", {-1, -1})) {
            ImPlot::SetupAxes("x", "y");
            ImPlot::SetupAxisLimits(
                ImAxis_X1,
                result.params.min_x,
                result.params.max_x,
                ImGuiCond_Always
            );
            ImPlot::SetupAxisLimits(
                ImAxis_Y1,
                result.params.min_y,
                result.params.max_y,
                ImGuiCond_Always
            );
            for (int i = 0; i < result.params.n_bodies; ++i) {
                const auto& track = result.tracks[i];
                ImPlotSpec spec;
                spec.LineColor = body_color(i);
                spec.LineWeight = 2.0f;
                ImPlot::PlotLine(
                    ("body " + std::to_string(i)).c_str(),
                    track.x.data(),
                    track.y.data(),
                    static_cast<int>(track.x.size()),
                    spec
                );
            }
            ImPlot::EndPlot();
        }
        ImGui::EndChild();
        ImGui::End();

        }
    );
}
