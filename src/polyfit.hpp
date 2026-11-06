#pragma once

#include <cstddef>
#include <glm/glm.hpp>
#include <span>

namespace poly {

struct Polynomial {
    float gradient;
    float intercept;

    Polynomial() = default;

    Polynomial(float gradient_, float intercept_) :
        gradient(gradient_),
        intercept(intercept_) {}

    inline float eval(const float dt) const {
        return (dt * this->gradient) + this->intercept;
    }
};

struct Polynomial3D {
    glm::vec3 gradients = {};
    glm::vec3 intercepts = {};

    Polynomial3D() = default;

    Polynomial3D(const glm::vec3& gradients_, const glm::vec3& intercepts_) :
        gradients(gradients_),
        intercepts(intercepts_) {}

    inline glm::vec3 eval(const float dt) const {
        return (dt * this->gradients) + this->intercepts;
    }
};

inline Polynomial
polyfit(std::span<const float> t, std::span<const float> val) {
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

    double sxx = 0.0f, sxy = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = t[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (val[i] - mean_y);
    }

    const float slope = static_cast<float>(sxy / sxx);
    const float gradient = static_cast<float>(mean_y - slope * mean_x);
    return Polynomial {slope, gradient};
}

inline Polynomial3D
polyfit3D(std::span<const float> t, std::span<const glm::vec3> vals) {
    const std::size_t n = t.size();
    if (n != vals.size())
        return {};
    if (n < 2)
        return {};

    glm::vec3 mean_x {}, mean_y {};
    for (std::size_t i = 0; i < n; ++i) {
        mean_x += t[i];
        mean_y += vals[i];
    }
    mean_x /= (float)n;
    mean_y /= (float)n;

    glm::vec3 sxx {}, sxy {};
    for (std::size_t i = 0; i < n; ++i) {
        const glm::vec3 dx = t[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (vals[i] - mean_y);
    }

    const auto slope = (sxy / sxx);
    const auto gradient = (mean_y - slope * mean_x);
    return Polynomial3D {slope, gradient};
}

template<typename T>
inline T vec_avg(std::span<const T> arr) {
    auto sum = arr[0] - arr[0];
    for (const auto& val : arr) {
        sum += val;
    }
    return sum / arr.size();
}

} // namespace poly
