// pybind11 module "filter_bindings" wrapping tracker.hpp.
//
//   import numpy as np, filter_bindings as pf
//   bank = pf.TrackerBank(pf.TrackerConfig(), threads=4)
//   bank.ingest_array(arr)      # arr: (N, 8) float64 = id, t, x, y, z, rc, vr, d
//   p = bank.predict(7, 12.5)   # p.x, p.y, p.z, p.sx ...
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "tracker.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

static_assert(
    sizeof(RadarItem) == 9 * sizeof(float),
    "RadarItem is not 9 floats"
);
static_assert(
    alignof(RadarItem) <= alignof(float),
    "RadarItem is over-aligned"
);
static_assert(std::is_trivially_copyable_v<RadarItem>);

PYBIND11_MAKE_OPAQUE(std::vector<Rock>);

namespace pybind11::detail {

template<>
struct type_caster<glm::vec3> {
    PYBIND11_TYPE_CASTER(
        glm::vec3,
        const_name("numpy.typing.NDArray[numpy.float32]")
    );

    bool load(handle src, bool convert) {
        if (!convert && !isinstance<array_t<float>>(src)) {
            return false;
        }
        auto arr =
            array_t<float, array::c_style | array::forcecast>::ensure(src);
        if (!arr) {
            PyErr_Clear();
            return false;
        }
        if (arr.ndim() != 1 || arr.shape(0) != 3) {
            return false;
        }
        value = glm::vec3(arr.at(0), arr.at(1), arr.at(2));
        return true;
    }

    static handle cast(const glm::vec3& v, return_value_policy, handle) {
        return array_t<float>(3, glm::value_ptr(v)).release();
    }
};
} // namespace pybind11::detail

using RadarArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

PYBIND11_MODULE(rock_tracker, m) {
    // Read-only view of a tracked rock (only what you get out of Tracker.rocks).
    py::class_<Rock>(m, "Rock")
        .def_readonly("position", &Rock::position)
        .def_readonly("diameter", &Rock::diameter)
        .def_readonly("velocity", &Rock::velocity)
        .def_readonly("last_seen", &Rock::last_seen);

    py::class_<Tracker>(m, "Tracker")
        .def(py::init<>())
        // Returns a snapshot (list of Rock copies), so it stays valid after update().
        .def_property_readonly(
            "rocks",
            [](const Tracker& t) { return t.rocks; }
        )
        .def(
            "update",
            [](Tracker& self,
               const RadarArray& radar_items,
               const glm::vec3& shuttle_position,
               float now) {
                // Accepted shapes (C-contiguous float32, converted if needed):
                //   (3, 3)     one item
                //   (N, 3, 3)  N items
                //   (N, 9)     N items, one flat row each
                // Row-major 3x3 layout of one item:
                //   [[x, y, z], [distance, range_rate, diameter], [time_stamp, pad, pad]]
                const auto nd = radar_items.ndim();
                const bool ok = radar_items.size() % 9 == 0
                    && ((nd == 2 && radar_items.shape(1) == 9)
                        || (nd == 2 && radar_items.shape(0) == 3
                            && radar_items.shape(1) == 3)
                        || (nd == 3 && radar_items.shape(1) == 3
                            && radar_items.shape(2) == 3));
                if (!ok) {
                    throw py::value_error(
                        "radar_items must have shape (3, 3), (N, 3, 3) or (N, 9)"
                    );
                }
                // Zero-copy view of the numpy buffer as RadarItems.
                const std::span<const RadarItem> items(
                    reinterpret_cast<const RadarItem*>(radar_items.data()),
                    static_cast<std::size_t>(radar_items.size() / 9)
                );
                self.update(items, shuttle_position, now);
            },
            "radar_items"_a,
            "shuttle_position"_a,
            "now"_a
        );
}
