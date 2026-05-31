cmake_minimum_required(VERSION 3.28.3)

project(fxfusion_engine
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

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)
find_package(Flatbuffers REQUIRED)

file(GLOB CPU_KERNEL_SOURCES csrc/kernels/cpu/*.cpp)
file(GLOB CUDA_KERNEL_SOURCES csrc/kernels/cuda/*.cpp)

add_library(fxfusion_extension MODULE
    csrc/runtime/execution_engine.cpp
    csrc/runtime/memory_manager.cpp
    csrc/runtime/op_registry.cpp
    csrc/bindings/python_bindings.cpp
    ${CPU_KERNEL_SOURCES}
    ${CUDA_KERNEL_SOURCES}
)

target_include_directories(fxfusion_extension PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/include
    ${CMAKE_SOURCE_DIR}/csrc/gen
)

target_link_libraries(fxfusion_extension PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Module
    flatbuffers::flatbuffers
)

target_compile_options(fxfusion_extension PRIVATE
    ${TORCH_CXX_FLAGS}
)

set_target_properties(fxfusion_extension PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/py/fxfusion/lib"
)