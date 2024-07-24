#include <cmath>

#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/stubdev/cuda_fullsolve.h"

namespace drake {
namespace pydrake {

void DefineCudaSAP(py::module m) {
  py::class_<FullSolveSAP>(m, "FullSolveSAP")
      .def(py::init<>())
      .def("init", &FullSolveSAP::init, py::arg("h_spheres"),
          py::arg("numProblems"), py::arg("numSpheres"), py::arg("numContacts"),
          py::arg("writeout"))
      .def("step", &FullSolveSAP::step)
      .def("destroy", &FullSolveSAP::destroy);
}

PYBIND11_MODULE(cuda_sap, m) {
  // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
  m.doc() = "Bindings for CUDA SAP.";
  DefineCudaSAP(m);
}

}  // namespace pydrake
}  // namespace drake
