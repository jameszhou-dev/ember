#include "../ember.h"

Sequential::Sequential(std::vector<Module> modules) {
    this->modules = modules;
}
void Sequential::zero_grad() {
    
}
void init_sequential(nb::module_& m) {
    nb::class_<Sequential>(m, "Sequential")
        .def(nb::init<std::vector<Module> &>());
}
