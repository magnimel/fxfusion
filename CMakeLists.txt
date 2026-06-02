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
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION} -- building with GPU support")
else()
    message(STATUS "CUDA not found -- building CPU-only")
endif()

file(GLOB CPU_KERNEL_SOURCES csrc/kernels/cpu/*.cpp)
file(GLOB CUDA_KERNEL_SOURCES csrc/kernels/cuda/*.cu)

if(CUDAToolkit_FOUND)
    set_source_files_properties(${CUDA_KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)
else()
    set(CUDA_KERNEL_SOURCES "")
endif()

add_library(fxfusion_extension MODULE
    csrc/runtime/execution_engine.cpp
    csrc/runtime/memory_manager.cpp
    csrc/runtime/op_registry.cpp
    csrc/bindings/python_bindings.cpp
    ${CPU_KERNEL_SOURCES}
    ${CUDA_KERNEL_SOURCES}
)

# CPU and runtime headers always present
target_include_directories(fxfusion_extension PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/include/runtime
    ${CMAKE_SOURCE_DIR}/csrc/include/kernels/cpu
    ${CMAKE_SOURCE_DIR}/csrc/gen
)

# CUDA kernel headers only when CUDA is available
if(CUDAToolkit_FOUND)
    target_include_directories(fxfusion_extension PRIVATE
        ${CMAKE_SOURCE_DIR}/csrc/include/kernels/cuda
    )
    target_compile_definitions(fxfusion_extension PRIVATE USE_CUDA)
endif()

target_link_libraries(fxfusion_extension PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Module
    flatbuffers::flatbuffers
)

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    target_link_libraries(fxfusion_extension PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(fxfusion_extension PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )
else()
    find_package(BLAS REQUIRED)
    target_link_libraries(fxfusion_extension PRIVATE ${BLAS_LIBRARIES})
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(fxfusion_extension PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_options(fxfusion_extension PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

target_compile_options(fxfusion_extension PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Rpass=loop-vectorize>
    ${TORCH_CXX_FLAGS}
)

set_target_properties(fxfusion_extension PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/py/fxfusion/lib"
)