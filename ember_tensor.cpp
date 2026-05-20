#include "ember.h"

std::vector<std::shared_ptr<Value>> Tensor::convert_vector(const std::vector<float>& input) {
    std::vector<std::shared_ptr<Value>> result;
    for (int i = 0; i < shape.second; i++) {
        result.push_back(std::make_shared<Value>(input[i]));
    }
    return result;
}
// sets the data and shape given a 1d vector
void Tensor::set_vars(const std::vector<float>& input) {
    data.clear();
    shape.first = 1;
    shape.second = input.size();
    data = convert_vector(input);
}
// sets the data and shape given a 2d vector
void Tensor::set_vars(const std::vector<std::vector<float>>& input) {
    data.clear();
    shape.first = input.size();
    shape.second = input[0].size();
    for (int i = 0; i < input.size(); i++) {
        std::vector<std::shared_ptr<Value>> temp = convert_vector(input[i]);
        data.insert(data.end(), temp.begin(), temp.end());
    }
}
// returns data in the form of a 2d vector
std::vector<std::vector<float>> Tensor::print_data() {
    std::vector<std::vector<float>> result;
    for (int i = 0; i < shape.first; i++) {
        std::vector<float> temp;
        for (int j = 0; j < shape.second; j++) {
            temp.push_back(data[i * shape.second + j]->data);  
        }
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
        .def("__pow__", [](Tensor &tensor, float exp) {
            Tensor result;
            result.shape = tensor.shape;
            for (int i = 0; i < tensor.data.size(); i++) {
                result.data.push_back(tensor.data[i]->pow(exp));
            }
            return result;
        })
        .def("__add__", [](Tensor &tensor, float num) {
            Tensor result;
            result.shape = tensor.shape;
            for (int i = 0; i < tensor.data.size(); i++) {
                result.data.push_back(tensor.data[i]->operator+(num));
            }
            return result;
        })
        .def("__sub__", [](Tensor &tensor, float num) {
            Tensor result;
            result.shape = tensor.shape;
            for (int i = 0; i < tensor.data.size(); i++) {
                result.data.push_back(tensor.data[i]->operator-(num));
            }
            return result;
        })
        .def("__mul__", [](Tensor &tensor, float num) {
            Tensor result;
            result.shape = tensor.shape;
            for (int i = 0; i < tensor.data.size(); i++) {
                result.data.push_back(tensor.data[i]->operator*(num));
            }
            return result;
        })
        .def("__truediv__", [](Tensor &tensor, float num) {
            Tensor result;
            result.shape = tensor.shape;
            for (int i = 0; i < tensor.data.size(); i++) {
                result.data.push_back(tensor.data[i]->operator/(num));
            }
            return result;
        })
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