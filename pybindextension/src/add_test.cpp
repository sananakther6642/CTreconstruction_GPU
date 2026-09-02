#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(py::array_t<double> array) {
    return 50;
}

PYBIND11_MODULE(add_test, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
