cmake_minimum_required(VERSION 3.15...4.2)
set(TARGET_NAME filtration-station)
set(CORE_NAME ${TARGET_NAME}-core)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES CXX)

option(STP_BUILD_GUI "Build the ImGui filter viewer" ON)
option(STP_BUILD_PYBINDINGS "Build the python bindings" OFF)
option(STP_ASAN "Build with address sanitize" OFF)
option(IMPLOT3D_STRICT_WARNINGS "Enable strict compiler warnings for ImPlot3D" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (MSVC)
    add_compile_options(/W4 /utf-8 /Ox)
else()
    set(DEBUG_FLAGS "-ggdb -rdynamic -Wextra")
    set(RELEASE_FLAGS "-Ofast")
    set(COMPILER_FLAGS "-Wall -Wno-unused -Werror=return-type ${DEBUG_FLAGS}")
    if (STP_ASAN)
        set(COMPILER_FLAGS "${COMPILER_FLAGS} -fsanitize=address")
    endif()
endif()
set(CMAKE_C_FLAGS ${COMPILER_FLAGS})
set(CMAKE_CXX_FLAGS  ${COMPILER_FLAGS})
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

if(MSVC)
else()
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()
find_package(Threads REQUIRED)
add_subdirectory(external/glm)

if(STP_BUILD_GUI)
    add_subdirectory(external/imgui)
    add_subdirectory(external/implot)
    add_subdirectory(external/implot3d)
endif()

if(STP_BUILD_PYBINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
        message(FATAL_ERROR
        "pybind11 submodule missing. Run: git submodule update --init --recursive")
    endif()
    add_subdirectory(external/pybind11)
endif()

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SOURCE_DIR}/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/replay_viewer\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/physics_viewer\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/physics\\.cpp$")

# --- core library: filters + tracker (no GUI, no Python) -----------------------------
add_library(${CORE_NAME} STATIC
  src/filters.cpp
  src/physics.cpp
)
target_include_directories(${CORE_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    external/pybind11/include
)
target_link_libraries(${CORE_NAME} PUBLIC glm::glm Threads::Threads)

if(STP_BUILD_PYBINDINGS)
    pybind11_add_module(rock_tracker MODULE src/bindings.cc)
    target_link_libraries(rock_tracker PRIVATE glm::glm Threads::Threads)
endif()

if(STP_BUILD_GUI)
    add_executable(${TARGET_NAME} ${SOURCE_FILES})
    target_link_libraries(${TARGET_NAME} PUBLIC imgui implot implot3d ${CORE_NAME})

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/replay_viewer.cpp")
        add_executable(stp-replay-viewer src/replay_viewer.cpp)
        target_link_libraries(stp-replay-viewer PUBLIC imgui implot implot3d ${CORE_NAME})
    endif()

    add_executable(stp-physics-viewer src/physics_viewer.cpp)
    target_link_libraries(stp-physics-viewer PUBLIC imgui implot implot3d ${CORE_NAME})
endif()
