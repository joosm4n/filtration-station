#include "physics.hpp"
#include "parallel.hpp"

#include <chrono>
#include <cmath>

namespace {

void resolve_walls(PhysicsBody& b, const PhysicsParams& p) {
    const float e = p.restitution;
    if (b.x - b.radius < p.min_x) {
        b.x = p.min_x + b.radius;
        b.vx = std::abs(b.vx) * e;
    } else if (b.x + b.radius > p.max_x) {
        b.x = p.max_x - b.radius;
        b.vx = -std::abs(b.vx) * e;
    }
    if (b.y - b.radius < p.min_y) {
        b.y = p.min_y + b.radius;
        b.vy = std::abs(b.vy) * e;
    } else if (b.y + b.radius > p.max_y) {
        b.y = p.max_y - b.radius;
        b.vy = -std::abs(b.vy) * e;
    }
}

void resolve_pair(PhysicsBody& a, PhysicsBody& b, float restitution) {
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float min_distance = a.radius + b.radius;
    const float distance_squared = dx * dx + dy * dy;
    if (distance_squared >= min_distance * min_distance)
        return;

    const float distance = std::sqrt(std::max(distance_squared, 1e-12f));
    const float nx = distance > 1e-6f ? dx / distance : 1.0f;
    const float ny = distance > 1e-6f ? dy / distance : 0.0f;
    const float overlap = min_distance - distance;
    const float inverse_a = 1.0f / a.mass, inverse_b = 1.0f / b.mass;
    const float inverse_total = inverse_a + inverse_b;
    a.x -= nx * overlap * inverse_a / inverse_total;
    a.y -= ny * overlap * inverse_a / inverse_total;
    b.x += nx * overlap * inverse_b / inverse_total;
    b.y += ny * overlap * inverse_b / inverse_total;

    const float relative_velocity = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
    if (relative_velocity >= 0.0f)
        return;
    const float impulse =
        -(1.0f + restitution) * relative_velocity / inverse_total;
    a.vx -= impulse * inverse_a * nx;
    a.vy -= impulse * inverse_a * ny;
    b.vx += impulse * inverse_b * nx;
    b.vy += impulse * inverse_b * ny;
}

} // namespace

PhysicsResult run_sim(const PhysicsParams& input) {
    PhysicsParams p = input;
    clamp_params(p);
    const int steps = std::max(1, static_cast<int>(std::ceil(p.duration / p.dt)));
    PhysicsResult result;
    result.params = p;
    result.t.reserve(steps + 1);
    result.tracks.resize(p.n_bodies);
    for (int i = 0; i < p.n_bodies; ++i) {
        auto& track = result.tracks[i];
        track.x.reserve(steps + 1);
        track.y.reserve(steps + 1);
        track.vx.reserve(steps + 1);
        track.vy.reserve(steps + 1);
    }

    std::array<PhysicsBody, PhysicsParams::MAX_BODIES> bodies = p.bodies;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step <= steps; ++step) {
        const double time = std::min<double>(step * p.dt, p.duration);
        result.t.push_back(time);
        for (int i = 0; i < p.n_bodies; ++i) {
            const auto& b = bodies[i];
            result.tracks[i].x.push_back(b.x);
            result.tracks[i].y.push_back(b.y);
            result.tracks[i].vx.push_back(b.vx);
            result.tracks[i].vy.push_back(b.vy);
        }
        if (step == steps)
            break;
        parallel_for(p.n_bodies, p.threads, [&](int i) {
            auto& b = bodies[i];
            b.vx = (b.vx + p.gravity_x * p.dt) * p.damping;
            b.vy = (b.vy + p.gravity_y * p.dt) * p.damping;
            b.x += b.vx * p.dt;
            b.y += b.vy * p.dt;
            resolve_walls(b, p);
        });
        for (int i = 0; i < p.n_bodies; ++i)
            for (int j = i + 1; j < p.n_bodies; ++j)
                resolve_pair(bodies[i], bodies[j], p.restitution);
    }
    result.compute_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start
                        )
                            .count();
    return result;
}
