#include <Python.h>
#include <torch/extension.h>
#include "runtime/execution_engine.hpp"

PYBIND11_MODULE(fxfusion_extension, m) {}

namespace fxfusion { 
    TORCH_LIBRARY(fxfusion_extension, m) {
        m.class_<ExecutionEngine>("ExecutionEngine")
        .def(torch::init<std::string, std::string>())
        .def("run", &ExecutionEngine::run);
    } 
}