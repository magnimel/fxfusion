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

    find_package(CUDNN REQUIRED)
    
    include(FetchContent)
    FetchContent_Declare(
        cudnn_frontend
        GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
        GIT_TAG v1.27.0
    )
    set(CUDNN_FRONTEND_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cudnn_frontend)

    
else()
    message(STATUS "CUDA not found -- building CPU-only")
endif()

file(GLOB CPU_KERNEL_SOURCES csrc/kernels/cpu/*.cpp)
file(GLOB RUNTIME_SOURCES csrc/runtime/*.cpp)

file(GLOB CUDA_KERNEL_SOURCES
    csrc/kernels/cuda/*.cu
    csrc/kernels/cuda/cache/*.cu
)

if(CUDAToolkit_FOUND)
    set_source_files_properties(${CUDA_KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)
else()
    set(CUDA_KERNEL_SOURCES "")
endif()

add_library(fxfusion_extension MODULE
    csrc/bindings/python_bindings.cpp
    ${RUNTIME_SOURCES}
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
        ${CMAKE_SOURCE_DIR}/csrc/include/kernels/cuda/internal
    )
    target_compile_definitions(fxfusion_extension PRIVATE USE_CUDA)
endif()

target_link_libraries(fxfusion_extension PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Module
    flatbuffers::flatbuffers
)

if(CUDAToolkit_FOUND)
    target_link_libraries(fxfusion_extension PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDNN::cudnn
        cudnn_frontend
    )
    target_compile_options(fxfusion_extension PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128,2810>
    )
endif()

target_compile_options(fxfusion_extension PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Rpass=loop-vectorize>
    ${TORCH_CXX_FLAGS}
)

if(CUDAToolkit_FOUND)
    set_target_properties(fxfusion_extension PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

set_target_properties(fxfusion_extension PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/py/fxfusion/lib"
)

add_custom_command(TARGET fxfusion_extension POST_BUILD
    COMMAND sh -c "du -sh \"$<TARGET_FILE:fxfusion_extension>\" | cut -f1 | sed 's/^/Output Library Size: /'"
    VERBATIM
)