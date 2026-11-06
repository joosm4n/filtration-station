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

#include <glm/ext/quaternion_geometric.hpp>
#include <glm/glm.hpp>

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
    glm::vec3 velocity = {};
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
        this->velocity = poly::polyfit3D(
                             this->history_times._data,
                             this->history_positions._data
        )
                             .eval(this->last_seen);
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
        if (glm::all(
                glm::equal(this->velocity, glm::vec3 {0.0f}, VEC3_EQ_EPSILON)
            )) {
            return this->position;
        }
        return this->position + this->velocity * (time - this->last_seen);
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
