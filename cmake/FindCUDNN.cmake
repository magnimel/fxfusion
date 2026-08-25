find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${CUDNN_PATH} $ENV{CUDNN_PATH} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include
)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${CUDNN_PATH} $ENV{CUDNN_PATH} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES lib lib64 lib/x64
)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_EXECUTABLE)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import nvidia.cudnn, os; print(list(nvidia.cudnn.__path__)[0]))"
            OUTPUT_VARIABLE PIP_CUDNN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PIP_CUDNN_DIR)
            find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS "${PIP_CUDNN_DIR}/include")
            find_library(CUDNN_LIBRARY NAMES cudnn HINTS "${PIP_CUDNN_DIR}/lib")
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)