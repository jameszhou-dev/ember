#include "../ember.h"


Module::Module() {
    this->num_inputs = -1;
    this->num_outputs = -1;
}

Module::Module(int num_inputs, int num_outputs) {
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;
    for (int i = 0; i < num_outputs; i++) {
        neurons.push_back(Neuron(num_inputs));
    }
}

std::vector<std::shared_ptr<Value>> Module::paramaters() {
    std::vector<std::shared_ptr<Value>> params;
    for (int i = 0; i < neurons.size(); i++) {
        std::vector<std::shared_ptr<Value>> temp;
        temp = neurons[i].parameters();
        params.insert(params.end(), temp.begin(), temp.end());
    }
    return params;
}

std::vector<std::shared_ptr<Value>> Module::forward(std::vector<std::shared_ptr<Value>> inputs) {
    std::vector<std::shared_ptr<Value>> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}

std::vector<std::shared_ptr<Value>> Module::forward(std::vector<float> inputs) {
    std::vector<std::shared_ptr<Value>> output;
    for (int i = 0; i < num_outputs; i++) {
        output.push_back(neurons[i].forward(inputs));
    }
    return output;
}

void init_module(nb::module_& m) {
    nb::class_<Module>(m, "Module");
}
