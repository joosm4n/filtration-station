cmake_minimum_required(VERSION 3.16)
project(implot CXX)

# Drop this file into the root of a manual implot clone (next to implot.cpp).
# Needs the `imgui` target to exist first (add_subdirectory(external/imgui) before this one).

add_library(implot STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/implot.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(implot PUBLIC cxx_std_17)
target_link_libraries(implot PUBLIC imgui)
