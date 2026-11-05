#pragma once

#include <algorithm>
#include <array>
#include <vector>

struct PhysicsBody {
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    float mass = 1.0f;
    float radius = 0.25f;
};

struct PhysicsParams {
    static constexpr int MAX_BODIES = 64;

    int n_bodies = 3;
    std::array<PhysicsBody, MAX_BODIES> bodies {{
        {-2.0f, 0.0f, 1.2f, 0.4f, 1.0f, 0.3f},
        {0.0f, 1.0f, -0.5f, -0.2f, 2.0f, 0.4f},
        {2.0f, -0.5f, -0.8f, 0.6f, 1.0f, 0.35f},
    }};

    float dt = 0.01f;
    float duration = 10.0f;
    float gravity_x = 0.0f;
    float gravity_y = -9.81f;
    float restitution = 0.8f;
    float damping = 0.999f;
    float min_x = -5.0f;
    float max_x = 5.0f;
    float min_y = -3.0f;
    float max_y = 3.0f;
    int threads = 1;
};

inline void clamp_params(PhysicsParams& p) {
    p.n_bodies = std::clamp(p.n_bodies, 1, PhysicsParams::MAX_BODIES);
    p.dt = std::clamp(p.dt, 0.0001f, 0.1f);
    p.duration = std::clamp(p.duration, p.dt, 10000.0f);
    p.restitution = std::clamp(p.restitution, 0.0f, 1.0f);
    p.damping = std::clamp(p.damping, 0.0f, 1.0f);
    p.threads = std::clamp(p.threads, 1, PhysicsParams::MAX_BODIES);
    if (p.max_x <= p.min_x)
        p.max_x = p.min_x + 1.0f;
    if (p.max_y <= p.min_y)
        p.max_y = p.min_y + 1.0f;
    for (auto& b : p.bodies) {
        b.mass = std::max(b.mass, 1e-4f);
        b.radius = std::clamp(b.radius, 0.001f, 1000.0f);
    }
}

struct PhysicsTrack {
    std::vector<double> x, y, vx, vy;
};

struct PhysicsResult {
    std::vector<double> t;
    std::vector<PhysicsTrack> tracks;
    double compute_ms = 0.0;
    PhysicsParams params;
};

PhysicsResult run_sim(const PhysicsParams& params);
