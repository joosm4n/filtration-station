cmake_minimum_required(VERSION 3.15...4.2)
set(TARGET_NAME filtration-station)
set(CORE_NAME ${TARGET_NAME}-core)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES CXX)

option(STP_BUILD_GUI "Build the ImGui filter viewer" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(COMPILER_FLAGS "-Wall -ggdb -Wno-unused -Werror=return-type -rdynamic -fsanitize=address")
# set(COMPILER_FLAGS "-Wall -ggdb -Wno-unused -Werror=return-type -rdynamic")
# set(COMPILER_FLAGS "-Wall -Ofast -Wno-unused -Werror=return-type")
set(CMAKE_C_FLAGS ${COMPILER_FLAGS})
set(CMAKE_CXX_FLAGS  ${COMPILER_FLAGS})
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

find_package(Threads REQUIRED)

add_subdirectory(external/glm)

if(STP_BUILD_GUI)
    add_subdirectory(external/imgui)
    add_subdirectory(external/implot)
endif()

set(PYBIND11_FINDPYTHON ON)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
    message(FATAL_ERROR
    "pybind11 submodule missing. Run: git submodule update --init --recursive")
endif()
add_subdirectory(external/pybind11)


set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SOURCE_DIR}/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/replay_viewer\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/physics_viewer\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*/physics\\.cpp$")

# --- core library: filters + tracker (no GUI, no Python) -----------------------------
add_library(${CORE_NAME} STATIC
  src/filters.cpp
  src/physics.cpp
  src/tracker.cpp
)
target_include_directories(${CORE_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    external/pybind11/include
)
target_link_libraries(${CORE_NAME} PUBLIC glm::glm Threads::Threads)

pybind11_add_module(filter_bindings src/bindings.cc)
target_link_libraries(filter_bindings PRIVATE ${CORE_NAME})

if(STP_BUILD_GUI)
    add_executable(${TARGET_NAME} ${SOURCE_FILES})
    target_link_libraries(${TARGET_NAME} PUBLIC imgui implot ${CORE_NAME})

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/replay_viewer.cpp")
        add_executable(stp-replay-viewer src/replay_viewer.cpp)
        target_link_libraries(stp-replay-viewer PUBLIC imgui implot ${CORE_NAME})
    endif()

    add_executable(stp-physics-viewer src/physics_viewer.cpp)
    target_link_libraries(stp-physics-viewer PUBLIC imgui implot ${CORE_NAME})
endif()
