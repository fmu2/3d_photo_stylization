#include "render.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize", &rasterize);
    m.def("refine_z_buffer", &refine_z_buffer);
    m.def("splat", &splat);
    m.def("splat_grad", &splat_grad);
}