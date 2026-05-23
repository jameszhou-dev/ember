#include "../ember.h"

std::vector<std::shared_ptr<Value>> Sequential::forward(std::vector<std::shared_ptr<Value>> inputs) {
    std::vector<std::shared_ptr<Value>> output = inputs;
    for (int i = 0; i < this->modules.size(); i++) {
        output = this->modules[i]->forward(output);
    }
    return output;
}

void Sequential::zero_grad() {
    for (int i = 0; i < this->modules.size(); i++) {
        this->modules[i]->zero_grad();
    }
}

void init_sequential(nb::module_& m) {
    nb::class_<Sequential>(m, "Sequential")
        .def("__init__", [](Sequential& self, nb::args args) {
            std::vector<nb::object> py_modules;
            for (size_t i = 0; i < args.size(); i++) {
                py_modules.push_back(nb::borrow(args[i]));  
            }
            new (&self) Sequential(std::move(py_modules));
        })
        .def("zero_grad", [](Sequential& self) {
            self.zero_grad();
        })
        .def("forward", [](Sequential& self, std::vector<std::vector<float>> inputs) {
            std::vector<std::vector<float>> results;
            for (auto& sample : inputs) {
                std::vector<std::shared_ptr<Value>> temp;
                for (float v : sample)
                    temp.push_back(Value::make(v));
                auto output = self.forward(temp);
                std::vector<float> result;
                for (auto& v : output)
                    result.push_back(v->data);
                results.push_back(result);
            }
            return results;
        })
        .def("forward", [](Sequential& self, std::vector<float> inputs) {
            std::vector<std::shared_ptr<Value>> temp;
            for (float v : inputs)
                temp.push_back(Value::make(v));
            auto output = self.forward(temp);
            std::vector<float> result;
            for (auto& v : output)
                result.push_back(v->data);
            return result;
        });
}