#pragma once

#include <cstddef>
#include <span>

namespace poly {

using Polynomial = std::pair<double, double>;

inline Polynomial
polyfit(std::span<const double> t, std::span<const double> val) {
    const std::size_t n = t.size();
    if (n != val.size())
        return {};
    if (n < 2)
        return {};

    double mean_x = 0.0, mean_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mean_x += t[i];
        mean_y += val[i];
    }
    mean_x /= (double)n;
    mean_y /= (double)n;

    double sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = t[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (val[i] - mean_y);
    }

    const double slope = sxy / sxx;
    return {slope, mean_y - slope * mean_x};
}

inline double eval(const Polynomial& pf, double dt) {
    return (dt * pf.first) + pf.second;
}

} // namespace poly

struct Polynomial3D {
    double mx, my, mz;
    double xint, yint, zint;

    Polynomial3D(
        std::span<const double> times,
        std::span<const double> x_vals,
        std::span<const double> y_vals,
        std::span<const double> z_vals
    ) {
        auto x = poly::polyfit(times, x_vals);
        auto y = poly::polyfit(times, y_vals);
        auto z = poly::polyfit(times, z_vals);
        this->mx = x.first;
        this->my = y.first;
        this->mz = z.first;
        this->xint = x.second;
        this->yint = y.second;
        this->zint = z.second;
    }
};
