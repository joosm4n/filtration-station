// pybind11 module "filter_bindings" wrapping tracker.hpp.
//
//   import numpy as np, filter_bindings as pf
//   bank = pf.TrackerBank(pf.TrackerConfig(), threads=4)
//   bank.ingest_array(arr)      # arr: (N, 8) = id, t, x, y, z, rc, vr, d
//   bank.ingest_radar_array(rows, ids, now, observer_position, observer_velocity)
//                              # rows: wso-stm (N, 3, 3) radar publication
//   p = bank.predict(7, 12.5)   # p.x, p.y, p.z, p.sx ...
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tracker.hpp"

namespace py = pybind11;

PYBIND11_MODULE(filter_bindings, m) {
    m.doc() = "Alpha-beta-gamma / Kalman object tracking";

    py::enum_<FilterType>(m, "FilterType")
        .value("ABG", FilterType::ABG)
        .value("KALMAN", FilterType::KALMAN);

    py::class_<SensorSpec>(m, "SensorSpec")
        .def(py::init<>())
        .def_readwrite("sig_pos", &SensorSpec::sig_pos)
        .def_readwrite("sig_rc", &SensorSpec::sig_rc)
        .def_readwrite("sig_vr", &SensorSpec::sig_vr)
        .def_readwrite("sig_diam_frac", &SensorSpec::sig_diam_frac);

    py::class_<TrackerConfig>(m, "TrackerConfig")
        .def(py::init<>())
        .def_readwrite("filter", &TrackerConfig::filter)
        .def_readwrite("alpha", &TrackerConfig::alpha)
        .def_readwrite("beta", &TrackerConfig::beta)
        .def_readwrite("gamma", &TrackerConfig::gamma)
        .def_readwrite("q", &TrackerConfig::q)
        .def_readwrite("p0", &TrackerConfig::p0)
        .def_readwrite("v0", &TrackerConfig::v0)
        .def_readwrite("sensor", &TrackerConfig::sensor);

    py::class_<Measurement>(m, "Measurement")
        .def(py::init<>())
        .def_readwrite("id", &Measurement::id)
        .def_readwrite("t", &Measurement::t)
        .def_readwrite("x", &Measurement::x)
        .def_readwrite("y", &Measurement::y)
        .def_readwrite("z", &Measurement::z)
        .def_readwrite("rc", &Measurement::rc)
        .def_readwrite("vr", &Measurement::vr)
        .def_readwrite("d", &Measurement::d);

    py::class_<Prediction>(m, "Prediction")
        .def_readonly("id", &Prediction::id)
        .def_readonly("t", &Prediction::t)
        .def_readonly("x", &Prediction::x)
        .def_readonly("y", &Prediction::y)
        .def_readonly("z", &Prediction::z)
        .def_readonly("vx", &Prediction::vx)
        .def_readonly("vy", &Prediction::vy)
        .def_readonly("vz", &Prediction::vz)
        .def_readonly("sx", &Prediction::sx)
        .def_readonly("sy", &Prediction::sy)
        .def_readonly("sz", &Prediction::sz);

    py::class_<TrackerBank>(m, "TrackerBank")
        .def(
            py::init<TrackerConfig, int>(),
            py::arg("config") = TrackerConfig {},
            py::arg("threads") = 1
        )
        .def("set_threads", &TrackerBank::set_threads)
        .def("ids", &TrackerBank::ids)
        .def("diameter", &TrackerBank::diameter)
        .def("reset", &TrackerBank::reset)
        .def(
            "ingest",
            &TrackerBank::ingest,
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "ingest_array",
            [](
                TrackerBank& b,
                py::array_t<double, py::array::c_style | py::array::forcecast> a
            ) {
                if (a.ndim() != 2 || a.shape(1) != 8)
                    throw std::invalid_argument(
                        "expected shape (N, 8): id, t, x, y, z, rc, vr, d"
                    );
                std::vector<Measurement> ms(a.shape(0));
                const double* r = a.data();
                for (py::ssize_t i = 0; i < a.shape(0); ++i, r += 8)
                    ms[i] = {
                        static_cast<int64_t>(r[0]),
                        r[1],
                        r[2],
                        r[3],
                        r[4],
                        r[5],
                        r[6],
                        r[7]
                    };
                py::gil_scoped_release
                    release; // heavy part runs without the GIL
                return b.ingest(ms);
            }
        )
        .def(
            "ingest_radar_array",
            [](
                TrackerBank& b,
                py::array_t<double, py::array::c_style | py::array::forcecast> rows,
                py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                double now,
                py::array_t<double, py::array::c_style | py::array::forcecast> observer_position,
                py::array_t<double, py::array::c_style | py::array::forcecast> observer_velocity,
                double clock_offset
            ) {
                if (rows.ndim() != 3 || rows.shape(1) != 3 || rows.shape(2) != 3)
                    throw std::invalid_argument("expected radar rows shape (N, 3, 3)");
                if (ids.ndim() != 1 || ids.shape(0) != rows.shape(0))
                    throw std::invalid_argument("expected ids shape (N,)");
                if (observer_position.ndim() != 1 || observer_position.shape(0) != 3
                    || observer_velocity.ndim() != 1 || observer_velocity.shape(0) != 3)
                    throw std::invalid_argument(
                        "expected observer position and velocity shape (3,)"
                    );

                const double* position = observer_position.data();
                const double* velocity = observer_velocity.data();
                const double* row = rows.data();
                const int64_t* id = ids.data();
                std::vector<Measurement> measurements(rows.shape(0));
                for (py::ssize_t i = 0; i < rows.shape(0); ++i, row += 9) {
                    const double source_time = row[6] + clock_offset;
                    const double lag = now - source_time;
                    measurements[i] = {
                        id[i],
                        source_time,
                        position[0] - lag * velocity[0] + row[0],
                        position[1] - lag * velocity[1] + row[1],
                        position[2] - lag * velocity[2] + row[2],
                        row[3],
                        row[4],
                        row[5]
                    };
                }
                py::gil_scoped_release release;
                return b.ingest(measurements);
            },
            py::arg("rows"),
            py::arg("ids"),
            py::arg("now"),
            py::arg("observer_position"),
            py::arg("observer_velocity"),
            py::arg("clock_offset") = 0.0
        )
        .def(
            "predict",
            &TrackerBank::predict,
            py::call_guard<py::gil_scoped_release>()
        )
        .def(
            "predict_all",
            &TrackerBank::predict_all,
            py::call_guard<py::gil_scoped_release>()
        );
}
