cmake_minimum_required(VERSION 3.30)
set(TARGET_NAME filtration-station)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES CXX)

find_package(Threads REQUIRED)

add_subdirectory(external/imgui)
add_subdirectory(external/implot)
add_subdirectory(external/glm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILER_FLAGS "-Wall -ggdb -Wno-unused -Werror=return-type -rdynamic")
# set(COMPILER_FLAGS "-Wall -Ofast -Wno-unused -Werror=return-type")
set(CMAKE_C_FLAGS ${COMPILER_FLAGS})
set(CMAKE_CXX_FLAGS  ${COMPILER_FLAGS})

# set(INCLUDES
#     ${CMAKE_CURRENT_SOURCE_DIR}/src
#     ${CMAKE_CURRENT_SOURCE_DIR}/vendored
# )
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} src)
file(GLOB_RECURSE SOURCE_FILES ${SOURCE_DIR}/*.cpp)
 
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC imgui implot glm::glm Threads::Threads)

