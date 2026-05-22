cmake_minimum_required(VERSION 3.28.3)

project(fxfusion
    VERSION 1.0.0
    DESCRIPTION "Execution Engine and C++/CUDA Operators"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Torch REQUIRED)

add_executable(engine
    src/engine.cpp
)

target_include_directories(engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/gen
)

target_link_libraries(engine PRIVATE
    ${TORCH_LIBRARIES}
)

target_compile_options(engine PRIVATE
    ${TORCH_CXX_FLAGS}
)
