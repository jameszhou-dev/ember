#include "../ember.h"

LinearLayer::LinearLayer(int num_inputs, int num_outputs) {
    Module(num_inputs, num_outputs);
}



void init_linear_layer(nb::module_& m) {
    nb::class_<LinearLayer, Module>(m, "Linear")
        .def(nb::init<const int&, const int&>());
}