#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

struct ReplayTrack {
    std::string id;
    std::vector<double> t, x, y, z, vx, vy, vz;
};

struct ReplayData {
    std::vector<double> radar_t, radar_x, radar_y;
    std::vector<ReplayTrack> tracks;
    std::string error;
};

static std::vector<double> numbers_between(const std::string& line, const std::string& key) {
    const auto begin = line.find(key);
    if (begin == std::string::npos)
        return {};
    const auto end = line.find("],\"tracks\"", begin);
    if (end == std::string::npos)
        return {};
    std::vector<double> values;
    const std::string source = line.substr(begin + key.size(), end - begin - key.size());
    const std::regex number_re(R"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
    for (std::sregex_iterator it(source.begin(), source.end(), number_re), finish;
         it != finish;
         ++it) {
        try {
            values.push_back(std::stod(it->str()));
        } catch (...) {
            return {};
        }
    }
    return values;
}

static bool load_replay(const char* path, ReplayData& data) {
    std::ifstream input(path);
    if (!input) {
        data.error = "Unable to open replay file";
        return false;
    }
    std::string line;
    const std::regex track_re(
        R"REGEX(\{"id":"([^"]+)","position_m":\[([^,]+),([^,]+),([^]]+)\],"velocity_mps":\[([^,]+),([^,]+),([^]]+)\])REGEX"
    );
    while (std::getline(input, line)) {
        if (line.find("\"type\":\"sample\"") == std::string::npos)
            continue;
        const auto time_pos = line.find("\"time_s\":");
        if (time_pos == std::string::npos) {
            data.error = "Sample is missing time_s";
            return false;
        }
        const double t = std::stod(line.substr(time_pos + 10));
        const auto radar = numbers_between(line, "\"radar_rows\":[");
        if (radar.size() % 9 != 0) {
            data.error = "radar_rows is not an array of 3x3 rows";
            return false;
        }
        for (size_t i = 0; i < radar.size(); i += 9) {
            data.radar_t.push_back(t);
            data.radar_x.push_back(radar[i]);
            data.radar_y.push_back(radar[i + 1]);
        }
        for (std::sregex_iterator it(line.begin(), line.end(), track_re), end; it != end; ++it) {
            const auto& match = *it;
            auto found = std::find_if(
                data.tracks.begin(), data.tracks.end(),
                [&](const ReplayTrack& track) { return track.id == match[1].str(); }
            );
            if (found == data.tracks.end()) {
                data.tracks.push_back(ReplayTrack {});
                found = std::prev(data.tracks.end());
                found->id = match[1].str();
            }
            found->t.push_back(t);
            found->x.push_back(std::stod(match[2].str()));
            found->y.push_back(std::stod(match[3].str()));
            found->z.push_back(std::stod(match[4].str()));
            found->vx.push_back(std::stod(match[5].str()));
            found->vy.push_back(std::stod(match[6].str()));
            found->vz.push_back(std::stod(match[7].str()));
        }
    }
    if (data.radar_t.empty() && data.tracks.empty()) {
        data.error = "Replay contains no samples";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    ReplayData data;
    if (!load_replay(argv[1], data))
        return 1;
    if (!glfwInit())
        return 1;
    GLFWwindow* window = glfwCreateWindow(1280, 800, "STP-C++ WSO-STM replay", nullptr, nullptr);
    if (!window)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    bool show_radar = true;
    std::vector<char> show(data.tracks.size(), 1);
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("WSO-STM replay");
        ImGui::Text("Tracks: %zu   Radar samples: %zu", data.tracks.size(), data.radar_t.size());
        ImGui::Checkbox("show radar", &show_radar);
        for (size_t i = 0; i < data.tracks.size(); ++i) {
            bool visible = show[i] != 0;
            if (ImGui::Checkbox(data.tracks[i].id.c_str(), &visible))
                show[i] = visible ? 1 : 0;
        }
        if (ImPlot::BeginPlot("Phobos-fixed XY", ImVec2(-1, -1))) {
            ImPlot::SetupAxes("x [m]", "y [m]");
            if (show_radar && !data.radar_t.empty())
                ImPlot::PlotScatter("radar", data.radar_x.data(), data.radar_y.data(), data.radar_t.size());
            for (size_t i = 0; i < data.tracks.size(); ++i) {
                if (show[i])
                    ImPlot::PlotLine(
                        data.tracks[i].id.c_str(),
                        data.tracks[i].x.data(),
                        data.tracks[i].y.data(),
                        data.tracks[i].x.size()
                    );
            }
            ImPlot::EndPlot();
        }
        ImGui::End();
        ImGui::Render();
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
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
