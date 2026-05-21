#ifndef EMBER_H
#define EMBER_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <string>
#include <vector>
#include <utility>
#include <iostream>
#include <functional>
#include <set>
#include <cmath>
#include <memory>
#include <fstream>  
#include <sstream>  
#include <algorithm>
#include <random>


namespace nb = nanobind;
using namespace nb::literals;

class Value : public std::enable_shared_from_this<Value> {  
public:
    float data;
    float grad = 0.0;
    std::vector<std::shared_ptr<Value>> prev;
    std::string label = "";
    std::string operation = "";
    std::function<void()> _backward = [](){};

    Value();
    Value(float data);
    Value(float data, std::string label);

    static std::shared_ptr<Value> make(float data, std::string label = "") {
        return std::make_shared<Value>(data, label);
    }
    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator+(float data);
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator*(float data);
    std::shared_ptr<Value> pow(float data);
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator/(float other);
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator-(float data);
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> node_exp();
    void backward();
    friend std::ostream& operator<<(std::ostream& os, const Value& value);
};

struct Tensor{
    std::vector<std::shared_ptr<Value>> data;
    std::pair<size_t, size_t> shape;

    std::vector<std::shared_ptr<Value>> convert_vector(const std::vector<float>& input);
    void set_vars(const std::vector<float>& input);
    void set_vars(const std::vector<std::vector<float>>& input);
    std::vector<std::vector<float>> print_data();
    void backward();
    
    Tensor();
    Tensor(const std::vector<float>& input);
    Tensor(const std::vector<std::vector<float>>& input);
};

void init_tensor(nb::module_& m);


#endif