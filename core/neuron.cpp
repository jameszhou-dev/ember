#include "../ember.h"

std::random_device rd;
std::mt19937 gen(67);
std::uniform_real_distribution<float> dist(-1.0, 1.0);

Neuron::Neuron(int num_inputs) {
    this->num_inputs = num_inputs;
    for (int i = 0; i < this->num_inputs; i++) {
        weights.push_back(Value::make(dist(gen)));
    }
    bias = Value::make(dist(gen));
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    params.insert(params.end(), weights.begin(), weights.end());
    params.push_back(bias);
    return params;
}
std::shared_ptr<Value> Neuron::forward(std::vector<std::shared_ptr<Value>> inputs) {
    std::shared_ptr<Value> sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        std::shared_ptr<Value> weighted_input = *weights[i] * inputs[i];
        sum = *sum + weighted_input;
    }
    return sum->tanh();
}

std::shared_ptr<Value> Neuron::forward(std::vector<float> inputs) {
    std::shared_ptr<Value> sum = bias;
    for (int i = 0; i < this->num_inputs; i++) {
        std::shared_ptr<Value> weighted_input = *weights[i] * inputs[i];
        sum = *sum + weighted_input;
    }
    return sum->tanh();
}


