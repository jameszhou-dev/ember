#include "ember.h"

Value::Value() {
    this->data = 0.0;
}

Value::Value(float data) {
    this->data = data;
}

Value::Value(float data, std::string label) {
    this->data = data;
    this->label = label;
}

std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> other) { // addition with another node
    auto out = std::make_shared<Value>(this->data + other->data);
    out->operation = "+";
    out->prev.push_back(shared_from_this());
    out->prev.push_back(other);
    out->_backward = [this, other, out]() mutable {
        this->grad += 1.0 * out->grad;
        other->grad += 1.0 * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator+(float data) { // addition with float
    return operator+(std::make_shared<Value>(data));
}

std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) { // multiplication with another node
    auto out = std::make_shared<Value>(this->data * other->data);
    out->operation = "*";
    out->prev.push_back(shared_from_this());
    out->prev.push_back(other);
    out->_backward = [this, other, out]() mutable {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator*(float data) { // multiplication with a float
    return operator*(std::make_shared<Value>(data));
}

std::shared_ptr<Value> Value::pow(float data) { // power with float
    auto out = std::make_shared<Value>(powf(this->data, data));
    out->operation = "^";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out, data]() mutable {
        this->grad += data * powf(this->data, data - 1) * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator/(std::shared_ptr<Value> other) { // division with another node
    return operator*(other->pow(-1));
}
std::shared_ptr<Value> Value::operator/(float other) {
    return operator*(std::make_shared<Value>(other)->pow(-1));
}
std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value> other) { // subtraction with another node
    return operator+(other->operator*(-1));
}

std::shared_ptr<Value> Value::operator-(float data) { // subtraction with another node
    return operator-(std::make_shared<Value>(data));
}

std::shared_ptr<Value> Value::tanh() {
    float x = this->data;
    float t = (expf(2*x) - 1) / (expf(2*x) + 1);
    auto out = std::make_shared<Value>(t);
    out->operation = "tanh";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out, t]() mutable {
        this->grad += (1 - powf(t, 2)) * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::node_exp() {
    float x = this->data;
    auto out = std::make_shared<Value>(expf(x));
    out->operation = "exp";
    out->prev.push_back(shared_from_this());
    out->_backward = [this, out]() mutable {
        this->grad += out->data * out->grad;
    };
    return out;
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::set<Value*> visited;
    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> node) {
        if (!visited.count(node.get())) {
            visited.insert(node.get());
            for (auto child : node->prev) {
                build_topo(child);
            }
            topo.push_back(node);
        }
    };
    build_topo(shared_from_this());
    this->grad = 1.0;
    for (int i = topo.size() - 1; i >= 0; i--) {
        topo[i]->_backward();
    }
}


std::ostream& operator<<(std::ostream& os, const Value& value) {
    os << "Value(data=" << value.data << ", grad=" << value.grad << ")";
    return os;
}
