#include "ember.h"


NB_MODULE(ember, m) {
    init_tensor(m);
    init_module(m);
    auto nn = m.def_submodule("nn");
    init_linear_layer(nn);
    init_sequential(nn);
}