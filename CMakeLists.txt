cmake_minimum_required(VERSION 3.15...4.2)
set(TARGET_NAME filtration-station)
set(CORE_NAME ${TARGET_NAME}-core)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(COMPILER_FLAGS "-Wall -ggdb -Wno-unused -Werror=return-type -rdynamic")
set(COMPILER_FLAGS "-Wall -Ofast -Wno-unused -Werror=return-type")
set(CMAKE_C_FLAGS ${COMPILER_FLAGS})
set(CMAKE_CXX_FLAGS  ${COMPILER_FLAGS})
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

find_package(Threads REQUIRED)

add_subdirectory(external/imgui)
add_subdirectory(external/implot)
add_subdirectory(external/glm)

set(PYBIND11_FINDPYTHON ON)
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11/CMakeLists.txt")
  message(FATAL_ERROR
    "pybind11 submodule missing. Run: git submodule update --init --recursive")
endif()
add_subdirectory(external/pybind11)


set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE SOURCE_FILES ${SOURCE_DIR}/*.cpp)

# --- core library: filters + tracker (no GUI, no Python) -----------------------------
add_library(${CORE_NAME} STATIC
  src/filters.cpp
  src/tracker.cpp
)
target_include_directories(${CORE_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    external/pybind11/include
)
target_link_libraries(${CORE_NAME} PUBLIC glm::glm Threads::Threads)

pybind11_add_module(filter_bindings src/bindings.cc)
target_link_libraries(filter_bindings PRIVATE ${CORE_NAME})
 
add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC imgui implot ${CORE_NAME})


