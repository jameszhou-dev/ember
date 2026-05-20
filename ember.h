#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <string>
#include <vector>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;


struct Tensor{
    std::vector<float> data;
    std::pair<size_t, size_t> shape;

    void set_vars(const std::vector<float>& input);
    void set_vars(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> print_data();

    Tensor();
    Tensor(const std::vector<float>& input);
    Tensor(const std::vector<std::vector<float>>& input);
};

void init_tensor(nb::module_& m);