// pybind11 module "filter_bindings" wrapping tracker.hpp.
//
//   import numpy as np, filter_bindings as pf
//   bank = pf.TrackerBank(pf.TrackerConfig(), threads=4)
//   bank.ingest_array(arr)      # arr: (N, 8) float64 = id, t, x, y, z, rc, vr, d
//   p = bank.predict(7, 12.5)   # p.x, p.y, p.z, p.sx ...

#include <cstdio>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "collision.hpp"
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
        .def_readonly("last_seen", &Rock::last_seen)
        .def("predicted_position", &Rock::predicted_position, "now"_a);

    py::class_<std::vector<Rock>>(m, "RockList")
        .def("__len__", [](const std::vector<Rock>& v) { return v.size(); })
        .def(
            "__genitem__",
            [](const std::vector<Rock>& v, py::ssize_t i) -> const Rock& {
                if (i < 0)
                    i += v.size();
                if (i < 0 || std::size_t(i) >= v.size())
                    throw py::index_error();
                return v[i];
            },
            py::return_value_policy::reference_internal
        )
        .def(
            "__iter__",
            [](const std::vector<Rock>& v) {
                return py::make_iterator(v.begin(), v.end());
            },
            py::keep_alive<0, 1>()
        );

    py::class_<Tracker>(m, "Tracker")
        .def(py::init<>())
        // Returns a snapshot (list of Rock copies), so it stays valid after update().
        .def_property_readonly(
            "rocks",
            [](Tracker& t) -> std::vector<Rock>& { return t.rocks; },
            py::return_value_policy::reference_internal
        )
        .def(
            "update",
            [](Tracker& self,
               const RadarArray& arr,
               const glm::vec3& shuttle_position,
               float now) {
                if (arr.size() == 0) {
                    printf("arr.size() == 0\n");
                    self.update({}, shuttle_position, now);
                }
                if (arr.ndim() != 3 || arr.shape(1) != 3 || arr.shape(2) != 3) {
                    throw py::value_error(
                        "expected a sequence of 3x3 matrices, i.e. shape (N, 3, 3)"
                    );
                }
                auto* first = reinterpret_cast<const RadarItem*>(arr.data());
                self.update(
                    std::span<const RadarItem>(
                        first,
                        static_cast<size_t>(arr.shape(0))
                    ),
                    shuttle_position,
                    now
                );
            }
        );

    py::class_<Avoider>(m, "Avoider")
        .def(py::init<>())
        .def(
            "choose",
            &Avoider::choose,
            "wanted_velocity"_a,
            "rocks"_a,
            "shuttle_position"_a,
            "now"_a
        );

    py::class_<BurnGovernor>(m, "BurnGovernor")
        .def(py::init<float>())
        .def(
            "acceleration",
            &BurnGovernor::acceleration,
            "wanted_velocity"_a,
            "current_velocity"_a
        );

    py::class_<GoalPlanner>(m, "GoalPlanner")
        .def(py::init<float>())
        .def(
            "miss_distance",
            &GoalPlanner::miss_distance,
            "shuttle_position"_a,
            "shuttle_velocity"_a,
            "target_position"_a
        )
        .def(
            "wanted_velocity",
            &GoalPlanner::wanted_velocity,
            "shuttle_position"_a,
            "shuttle_velocity"_a,
            "target_position"_a
        );

    using namespace collision;
    py::class_<Cone>(m, "Cone")
        .def(py::init<>())
        .def("reset", &Cone::reset, "start_point"_a, "end_point"_a)
        .def("set_angle_degrees", &Cone::set_angle_degrees, "ang_degrees"_a)
        .def("check_collision", &Cone::check_collision, "point"_a);
}
