#include "../ember.h"

Module::~Module() {}

void init_module(nb::module_& m) {
    nb::class_<Module>(m, "Module");
}
