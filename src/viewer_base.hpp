#pragma once

#include <GLFW/glfw3.h>
#include <cstdio>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>

struct ViewerConfig {
    const char* title;
    int width;
    int height;
    const char* ini_filename;
    ImVec4 clear_color;
};

struct RunControls {
    bool run_requested = false;
    bool auto_run_changed = false;
};

class GenericViewer {
  public:
    explicit GenericViewer(ViewerConfig config) : config_(config) {}

    RunControls drawRunControls(bool& auto_run, bool busy) const {
        RunControls controls;
        ImGui::SeparatorText("Run");
        controls.auto_run_changed = ImGui::Checkbox("auto rerun", &auto_run);
        controls.run_requested = ImGui::Button("Run", {-1, 0});
        ImGui::TextDisabled(busy ? "computing..." : "chilling...");
        return controls;
    }

    template<class Setup, class Loaded, class Frame>
    int run(Setup&& setup, Loaded&& loaded, Frame&& frame) {
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
        if (!glfwInit()) {
            const char* err_msg = NULL;
            glfwGetError(&err_msg);
            printf("[error] glfw failed to init - %s\n", err_msg);
            return 1;
        }
        GLFWwindow* window = glfwCreateWindow(
            config_.width,
            config_.height,
            config_.title,
            nullptr,
            nullptr
        );
        if (!window) {
            glfwTerminate();
            return 1;
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        printf("[setup]\n");
        setup();
        ImGui::GetIO().IniFilename = config_.ini_filename;
        ImGui::LoadIniSettingsFromDisk(ImGui::GetIO().IniFilename);

        printf("[loaded]\n");
        loaded();

        printf("[main loop]\n");
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            frame();

            ImGui::Render();
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            glViewport(0, 0, width, height);
            glClearColor(
                config_.clear_color.x,
                config_.clear_color.y,
                config_.clear_color.z,
                config_.clear_color.w
            );
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

  private:
    ViewerConfig config_;
};
