#include "ember.h"

// sets the data and shape given a 1d vector
void Tensor::set_vars(const std::vector<float>& input) {
    data.clear();
    shape.first = 1;
    shape.second = input.size();
    data = input;
}
// sets the data and shape given a 2d vector
void Tensor::set_vars(const std::vector<std::vector<float>>& input) {
    data.clear();
    shape.first = input.size();
    shape.second = input[0].size();
    for (int i = 0; i < input.size(); i++) {
        data.insert(data.end(), input[i].begin(), input[i].end());
    }
}
// returns data in the form of a 2d vector
std::vector<std::vector<float>> Tensor::print_data() {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < shape.first ; i++) {
        std::vector<float> temp(data.begin() + (i*shape.second), data.begin() + (i*shape.second) + shape.second);
        result.push_back(temp);
    }
    return result;
}

Tensor::Tensor() {}
Tensor::Tensor(const std::vector<float>& input) {
    set_vars(input);
}
Tensor::Tensor(const std::vector<std::vector<float>>& input) {
    set_vars(input);
}

void init_tensor(nb::module_& m) {
    nb::class_<Tensor>(m, "Tensor")
        .def(nb::init<>())
        .def(nb::init<const std::vector<float> &>())
        .def(nb::init<const std::vector<std::vector<float>> &>())
        .def_rw("shape", &Tensor::shape)
        .def_prop_rw("data", 
                    [] (Tensor& tensor)  {
                        return tensor.print_data();
                    },
                    [] (Tensor& tensor, nb::object input) {
                        try {
                            auto vec2d = nb::cast<std::vector<std::vector<float>>>(input);
                            tensor.set_vars(vec2d);
                        } catch (const nb::cast_error&) {
                            auto vec1d = nb::cast<std::vector<float>>(input);
                            tensor.set_vars(vec1d);
                        }
                    });
}