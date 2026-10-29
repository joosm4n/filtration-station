#pragma once
//
// #include <algorithm>
// #include <chrono>
// #include <cstddef>
// #include <thread>
// #include <vector>
//
// // Anything with std::vector<double> members x_k, v_k, a_k works here
// // (adapt the field names / types to your filter class).
// template<typename Filter>
// void live_plot(
//     const Filter& f,
//     const std::vector<double>& zs,
//     double dt,
//     double pause_s = 0.05,
//     const std::vector<double>& truth = {}
// ) { // true position, aligned with zs
//     using namespace matplot;
//
//     // Persistent state across calls (equivalent of the hasattr(live_plot, ...) trick)
//     static figure_handle fig;
//     static std::vector<axes_handle> axs;
//
//     if (!fig) { // first call: create the figure
//         fig = figure(true); // true = quiet mode (no automatic redraws)
//         fig->size(700, 700);
//         tiledlayout(3, 1); // returns void; configures the current figure
//         for (int i = 0; i < 3; ++i)
//             axs.push_back(nexttile());
//     }
//
//     // time axis
//     std::vector<double> t(f.x_k.size());
//     for (std::size_t i = 0; i < t.size(); ++i)
//         t[i] = static_cast<double>(i) * dt;
//
//     struct Series {
//         const char* name;
//         const std::vector<double>* data;
//     };
//
//     const Series series[] =
//         {{"position", &f.x_k}, {"velocity", &f.v_k}, {"accel", &f.a_k}};
//
//     for (int i = 0; i < 3; ++i) {
//         auto& ax = axs[i];
//         ax->clear(); // ax.clear()
//         ax->hold(on);
//
//         auto p = ax->plot(t, *series[i].data, "-o");
//         p->marker_size(3);
//         p->display_name("estimate");
//
//         ax->ylabel(series[i].name);
//         if (!t.empty())
//             ax->xlim({t.front(), t.back()}); // sharex=True (approximation)
//     }
//
//     // measurements: zs lines up with t[1:]
//     if (!zs.empty() && t.size() > 1) {
//         std::vector<double> t_meas(t.begin() + 1, t.end());
//         t_meas.resize(std::min(t_meas.size(), zs.size()));
//         std::vector<double> z(zs.begin(), zs.begin() + t_meas.size());
//
//         auto m = axs[0]->plot(t_meas, z, "x");
//         m->color({0.5f, 0.5f, 0.5f}); // gray
//         m->display_name("measurement");
//     }
//
//     // actual (noise-free) position, also aligned with t[1:]
//     if (!truth.empty() && t.size() > 1) {
//         std::vector<double> t_true(t.begin() + 1, t.end());
//         t_true.resize(std::min(t_true.size(), truth.size()));
//         std::vector<double> y(truth.begin(), truth.begin() + t_true.size());
//
//         auto a = axs[0]->plot(t_true, y, "k--");
//         a->display_name("actual");
//     }
//
//     auto lg = legend(axs[0], {});
//     lg->location(legend::general_alignment::topleft);
//
//     axs[2]->xlabel("time [s]");
//
//     // gnuplot >= 6.0.1 warns about Matplot++'s inline ('-') data inside a multiplot
//     // on every redraw (Matplot++ issue #432). Silence it; the data is still read.
//     fig->backend()->run_command("unset warnings");
//
//     fig->draw(); // redraw
//     if (pause_s > 0.0)
//         std::this_thread::sleep_for(
//             std::chrono::duration<double>(pause_s)
//         ); // plt.pause
// }
//
// // 2D live plot for two independent alpha-beta-gamma filters (one per axis).
// // Filter needs std::vector<double> members x_k, v_k, a_k.
// //
// // zx/zy are the noisy measurements and truth_x/truth_y the actual positions; all
// // four line up with the filter states at t[1:] (one entry per predict/update step).
// //
// // Layout (2x2):   [ x-y path        ] [ position vs time     ]
// //                 [ velocity vs time ] [ acceleration vs time ]
// template<typename Filter>
// void live_plot_2d(
//     const Filter& fx,
//     const Filter& fy,
//     const std::vector<double>& zx,
//     const std::vector<double>& zy,
//     const std::vector<double>& truth_x,
//     const std::vector<double>& truth_y,
//     double dt,
//     double pause_s = 0.0
// ) {
//     using namespace matplot;
//
//     static figure_handle fig;
//     static std::vector<axes_handle> axs;
//
//     if (!fig) { // first call: create the figure
//         fig = figure(true); // true = quiet mode (no automatic redraws)
//         fig->size(900, 800);
//         tiledlayout(2, 2); // returns void; configures the current figure
//         for (int i = 0; i < 4; ++i)
//             axs.push_back(nexttile());
//     }
//
//     const std::array<float, 3> cx {0.12f, 0.47f, 0.71f}; // x axis: blue
//     const std::array<float, 3> cy {0.85f, 0.33f, 0.10f}; // y axis: orange
//     const std::array<float, 3> grey {0.5f, 0.5f, 0.5f};
//
//     const std::size_t n = std::min(fx.x_k.size(), fy.x_k.size());
//     std::vector<double> t(n);
//     for (std::size_t i = 0; i < n; ++i)
//         t[i] = static_cast<double>(i) * dt;
//
//     // number of step entries (measurements/truth) that line up with t[1:]
//     std::size_t m = 0;
//     if (n > 1) {
//         m = std::min(
//             {n - 1, zx.size(), zy.size(), truth_x.size(), truth_y.size()}
//         );
//     }
//     auto head = [m](const std::vector<double>& v) {
//         return std::vector<double>(v.begin(), v.begin() + m);
//     };
//     const std::vector<double> t_tail(
//         t.begin() + (n > 0 ? 1 : 0),
//         t.begin() + (n > 0 ? 1 : 0) + m
//     );
//
//     // ---- tile 0: x-y path --------------------------------------------------
//     {
//         auto& ax = axs[0];
//         ax->clear();
//         ax->hold(on);
//
//         if (m > 0) {
//             auto actual = ax->plot(head(truth_x), head(truth_y), "k--");
//             actual->display_name("actual");
//
//             auto meas = ax->plot(head(zx), head(zy), "x");
//             meas->color(grey);
//             meas->display_name("measurement");
//         }
//         std::vector<double> ex(fx.x_k.begin(), fx.x_k.begin() + n);
//         std::vector<double> ey(fy.x_k.begin(), fy.x_k.begin() + n);
//         auto est = ax->plot(ex, ey, "-o");
//         est->marker_size(3);
//         est->display_name("estimate");
//
//         ax->xlabel("x");
//         ax->ylabel("y");
//         auto lg = legend(ax, {});
//         lg->location(legend::general_alignment::topleft);
//     }
//
//     // ---- time-series tiles ------------------------------------------------
//     auto series_tile = [&](axes_handle& ax,
//                            const char* ylabel,
//                            const std::vector<double>& dx,
//                            const std::vector<double>& dy) {
//         ax->clear();
//         ax->hold(on);
//
//         std::vector<double> vx(dx.begin(), dx.begin() + n);
//         std::vector<double> vy(dy.begin(), dy.begin() + n);
//
//         auto px = ax->plot(t, vx, "-o");
//         px->marker_size(3);
//         px->color(cx);
//         px->display_name("x");
//
//         auto py = ax->plot(t, vy, "-o");
//         py->marker_size(3);
//         py->color(cy);
//         py->display_name("y");
//
//         ax->ylabel(ylabel);
//         if (t.size() > 1)
//             ax->xlim({t.front(), t.back()});
//     };
//
//     series_tile(axs[1], "position", fx.x_k, fy.x_k);
//     series_tile(axs[2], "velocity", fx.v_k, fy.v_k);
//     series_tile(axs[3], "accel", fx.a_k, fy.a_k);
//
//     // actual position per axis (dashed, same colour as the estimate)
//     if (m > 0) {
//         auto ax_x = axs[1]->plot(t_tail, head(truth_x), "--");
//         ax_x->color(cx);
//         ax_x->display_name("x actual");
//         auto ax_y = axs[1]->plot(t_tail, head(truth_y), "--");
//         ax_y->color(cy);
//         ax_y->display_name("y actual");
//     }
//     auto lg = legend(axs[1], {});
//     lg->location(legend::general_alignment::topleft);
//
//     axs[2]->xlabel("time [s]");
//     axs[3]->xlabel("time [s]");
//
//     // gnuplot >= 6.0.1 warns about Matplot++'s inline ('-') data inside a multiplot
//     // on every redraw (Matplot++ issue #432). Silence it; the data is still read.
//     fig->backend()->run_command("unset warnings");
//
//     fig->draw();
//     if (pause_s > 0.0)
//         std::this_thread::sleep_for(std::chrono::duration<double>(pause_s));
// }
