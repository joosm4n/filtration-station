#pragma once
// Online tracking API on top of filters.hpp.
//
//   TrackerBank bank(cfg, /*threads=*/4);
//   bank.ingest(measurements);                 // any number of objects, any order
//   Prediction p = bank.predict(id, t_future); // position (and velocity) at time t
//
// Design rules (so it can be wrapped for Python / used from many threads):
//   * public types are plain structs of double / int64 / enum, no glm, no unique_ptr
//   * Tracker      : one object, NOT thread-safe on its own
//   * TrackerBank  : thread-safe (shared_mutex); ingest() runs objects in parallel
//   * nothing throws inside worker threads; errors are thrown on the calling thread
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>
#include <optional>
#include <limits>
#include <ranges>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/gtx/vector_space.hpp>

#include "polyfit.hpp"

// #define self (*this)

constexpr glm::vec3 VEC3_EQ_EPSILON = glm::vec3 {0.0001f};

template<typename T, std::size_t N>
struct RingBuffer {
    std::array<T, N> _data = {};
    std::size_t _idx = 0;
    bool filled = false;

    RingBuffer() = default;

    inline void push_back(const T& item) {
        this->_data[this->_idx++] = item;
        if (this->_idx == N) {
            filled = true;
            this->_idx = 0;
        }
    }

    inline const T& back() const {
        return this->_data[this->_idx];
    }

    inline const T& front() {
        if (filled) {
            return this->_data[(this->_idx + 1) % N];
        }
        return this->_data[0];
    }

    inline std::size_t size() const {
        if (filled)
            return N;
        return this->_idx;
    }
};

struct HistoryItem {
    float time;
    glm::vec3 pos;
};

struct Rock {
    glm::vec3 position = {};
    float diameter = 0.0f;
    std::optional<glm::vec3> velocity = {};
    float last_seen = 0.0f;

    static constexpr std::size_t HISTORY_LEN = 30;
    RingBuffer<float, HISTORY_LEN> history_times = {};
    RingBuffer<glm::vec3, HISTORY_LEN> history_positions = {};

    Rock(const glm::vec3& pos, float diameter_, float time) :
        position(pos),
        diameter(diameter_),
        last_seen(time) {
        history_times.push_back(time);
        history_positions.push_back(position);
    }

    inline void update_velocity() {
        constexpr float MIN_TIME_STEP = 1.0f;
        if (this->history_times.size() < 3
            || this->history_times.back() - this->history_times.front()
                < MIN_TIME_STEP) {
            return;
        }
        this->velocity.emplace(poly::polyfit3D(                             
            this->history_times._data,
            this->history_positions._data
        ).eval(this->last_seen));
    }

    inline void
    add_sighting(const glm::vec3& pos, float diameter_, float time) {
        if (time <= this->last_seen) {
            return;
        }
        this->position = pos;
        this->diameter = diameter_;
        this->last_seen = time;
        this->history_times.push_back(time);
        this->history_positions.push_back(pos);
    }

    inline glm::vec3 predicted_position(float time) const {
        if (this->velocity.has_value()) {
            const auto vel = this->velocity.value(); 
            return this->position + vel * (time - this->last_seen);
        } else {
            return this->position;
        }
    }

    inline glm::vec3 vel_or() const {
        return this->velocity.value_or(glm::vec3{0.0f});
    }
};

struct RadarItem {
    glm::vec3 position = {};
    float distance = {};
    float range_rate = {};
    float diameter = {};
    float time_stamp = {};

  private:
    const float _padding1 = 0.0f;
    const float _padding2 = 0.0f;
};

struct Tracker {
    std::vector<Rock> rocks = {};
    float gate = 15.0f;
    float max_age = 5.0f;

    static constexpr std::size_t MIN_WORKER_PER_THREAD = 20'000;

    template<typename F>
    static void parallel_for(std::size_t n, std::size_t min_chunk, F&& fn) {
        const std::size_t hw =
            std::max(1u, std::thread::hardware_concurrency());
        const std::size_t nthreads = std::min(
            hw,
            (n + min_chunk - 1) / std::max<std::size_t>(min_chunk, 1)
        );
        if (nthreads <= 1) {
            fn(std::size_t {0}, n);
            return;
        }
        const std::size_t chunk = (n + nthreads - 1) / nthreads;
        std::vector<std::jthread> workers;
        workers.reserve(nthreads - 1);
        for (std::size_t t = 1; t < nthreads; ++t) {
            const std::size_t lo = t * chunk;
            const std::size_t hi = std::min(n, lo + chunk);
            if (lo >= hi)
                break;
            workers.emplace_back([&fn, lo, hi] { fn(lo, hi); });
        }
        fn(std::size_t {0}, std::min(n, chunk));
    } // ~jthread joins here

    std::ptrdiff_t
    find_match(const glm::vec3& position, float diameter, float time) const {
        float best_d2 = this->gate * this->gate;
        std::ptrdiff_t best = -1;
        for (std::size_t i = 0; i < this->rocks.size(); ++i) {
            const auto& rock = this->rocks.at(i);
            if (std::abs(rock.diameter - diameter) > 0.25f * rock.diameter) {
                continue; // sizes too different
            }
            const auto d = rock.predicted_position(time) - position;
            const float d2 = glm::dot(d, d);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = static_cast<std::ptrdiff_t>(i);
            }
        }
        return best;
    }

    void update(
        const std::span<const RadarItem> radar_items,
        const glm::vec3& shuttle_position,
        float now
    ) {
        const std::size_t n = radar_items.size();
        std::vector<std::ptrdiff_t> match(n, -1);

        const std::size_t min_chunk = std::max<std::size_t>(
            1,
            MIN_WORKER_PER_THREAD / std::max<std::size_t>(1, this->rocks.size())
        );

        parallel_for(n, min_chunk, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i) {
                const auto& item = radar_items[i];
                match[i] = this->find_match(
                    shuttle_position + item.position,
                    item.diameter,
                    item.time_stamp
                );
            }
        });
        for (std::size_t i = 0; i < n; ++i) {
            const auto& item = radar_items[i];
            const glm::vec3 position = shuttle_position + item.position;
            if (match[i] < 0) {
                this->rocks
                    .emplace_back(position, item.diameter, item.time_stamp);
            } else {
                this->rocks[static_cast<std::size_t>(match[i])]
                    .add_sighting(position, item.diameter, item.time_stamp);
            }
        }

        std::erase_if(this->rocks, [&](const Rock& r) {
            return now - r.last_seen >= this->max_age;
        });
    }
};

// If we fly at my_velocity, how close do we get to the worst rock in the next `horizon` seconds,
// after subtracting its keep-out radius? Positive = safe, negative = collision course.
// rel_pos  (n,3)  where each rock is, relative to us
// rock_vel (n,3)  how fast each rock is moving
// keep_out (n,)   radius we want to stay outside (rock radius + safety margin)
inline float clearance(const glm::vec3& my_velocity, std::span<const glm::vec3> rel_pos, std::span<const glm::vec3> rock_vel, std::span<const float> keep_out, float horizon=15.0) {
    float worst = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < rel_pos.size(); ++i) {
        glm::vec3 rel_vel = rock_vel[i] - my_velocity;
        float speed_squared = glm::dot(rel_vel, rel_vel) + 1e-9f;

        // time of closest approach, clamped to [0, horizon]
        float t = -glm::dot(rel_pos[i], rel_vel) / speed_squared;
        t = std::clamp(t, 0.0f, horizon);

        float closest = glm::length(rel_pos[i] + t * rel_vel);
        worst = std::min(worst, closest - keep_out[i]);
    }
    return worst;
}

inline std::pair<float, float> two_sideways_directions(const auto& direction) {
    // Two unit vectors that are perpendicular to `direction` and to each other.
    glm::vec3 helper = glm::abs(direction[2]) < 0.9f ?
        glm::vec3{0.0f, 0.0f, 1.0f} :
        glm::vec3{1.0f, 0.0f, 0.0f};

    const auto side1 = glm::cross(direction, helper);
    const auto side1 /= glm::length(side1);
    const auto side2 = glm::cross(direction, side1);
    return {side1, side2};
}

template<std::size_t N, typename T= float>
consteval std::array<T, N> linspace(T start, T stop, bool endpoint = true) {
    std::array<T, N> arr{};
    if (endpoint) {
        const auto step = 
    }
    for (std::size_t i = 0; i < N; ++i) {
        
    }
    return arr;
}

struct Avoider {
    float margin = 6.0f;    // metres of clearance beyond each rock's surface
    float unknown_extra = 4.0f; // extra margin for rocks whose velocity we don't know yet
    float horizon = 50.0f;
    std::optional<glm::vec3> last_dodge = std::nullopt; // remembered so we don't flip-flop between dodges

    inline glm::vec3 choose(const glm::vec3& wanted_velocity, const std::vector<Rock>& rocks, const glm::vec3& shuttle_position, float now) {
        // Return the velocity to fly: the wanted one if it is safe, else the smallest safe change.
        const float speed = glm::length(wanted_velocity);
        if (speed < 1e-3f || rocks.size() == 0) {
            return wanted_velocity;
        }
        const std::size_t n_rocks = rocks.size();

        // Describe every rock as plain arrays.
        auto rel_pos_view = rocks | std::views::transform([&](const Rock& r) {
            return r.predicted_position(now) - shuttle_position;
        });
        const auto rel_pos = std::vector(rel_pos_view.begin(), rel_pos_view.end());

        auto rock_vel_view = rocks | std::views::transform([&](const Rock& r) {
            return r.vel_or();
        });
        const auto rock_vel = std::vector(rock_vel_view.begin(), rock_vel_view.end());

        auto keep_out_view = rocks | std::views::transform([&](const Rock& r) {
            return r.diameter / 2 + this->margin + (r.velocity.has_value() ? this->unknown_extra : 0.0f);
        });
        const auto keep_out = std::vector(keep_out_view.begin(), keep_out_view.end());

        auto score = [&](const glm::vec3& velocity) {
            return clearance(velocity, rel_pos, rock_vel, keep_out, this->horizon);
        };

        // 1. Nothing in the way? Fly exactly as wanted (no thruster use at all).
        if (score(wanted_velocity) > 0.0f) {
            this->last_dodge = std::nullopt;
            return wanted_velocity;
        }

        // 2. Still dodging? Keep the same dodge while it still works (stops flip-flopping).
        if (this->last_dodge.has_value()) {
            if (score(this->last_dodge.value()) > 0) 
            return this->last_dodge.value();
        }

        // 3. Try sideways dodges, SMALLEST first: a small velocity change costs the least fuel.
        auto [side1, side2] = two_sideways_directions(wanted_velocity / speed);
        auto tried = [wanted_velocity];

        constexpr auto angles = (0, 2 * glm::pi, 8);
        for (const auto& size : {0.05, 0.1, 0.2, 0.35, 0.5, 0.75}){         // as a fraction of our speed
            for (const auto angle : np.linspace(0, 2 * np.pi, 8, endpoint=False)) {
                sideways = np.cos(angle) * side1 + np.sin(angle) * side2;
                candidate = wanted_velocity + size * speed * sideways;
                if (score(candidate) > 0){
                    self.last_dodge = candidate;
                    return candidate;
                }
                tried.append(candidate);
            }
        }

        // 4. No sideways dodge works: try slowing down.
        for fraction in (0.5, 0.25, 0.0):
            candidate = wanted_velocity * fraction
            if score(candidate) > 0:
                self.last_dodge = candidate
                return candidate
            tried.append(candidate)

        // 5. Nothing is fully safe: take the least-bad option we looked at.
        self.last_dodge = max(tried, key=score)
        return self.last_dodge
    }
}