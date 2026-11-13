#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sys/types.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/trigonometric.hpp>

#include <format>
#include <ranges>
#include <span>
#include <sstream>
#include <vector>

#include "nd_array.hpp"

/*
 * Plan:
 *  Create a frustum
 *      - Get a 2 param func to define cone from line from ship to station
 *      - Set a slope (radius rate)
 *
 *      angle between vectors 'u' and 'v' = acos( (u * v) / (|u| * |v|) )
 *
 *
 *  What can we ignore? 
 *      Anything moving away from us
 *      
 */

namespace collision {

inline float vec_angle(const glm::vec3& a, const glm::vec3& b) {
    const auto dot = glm::dot(a, b);
    const auto ang = glm::acos(dot / (glm::length(a) * glm::length(b)));
    return ang;
}

constexpr std::size_t AVERAGE_RUNTIME_SECS = 200; // approx
constexpr std::size_t MAX_VELOCITY = 5; // approx (m/s)

struct Cone {
    glm::vec3 start_position;
    glm::vec3 end_position;
    glm::vec3 raw_vector;
    glm::vec3 vec_norm;
    float vec_len;
    float angle_rads = glm::radians(45.0f);

    struct Mesh {
        std::vector<glm::vec3> verts;
        std::vector<unsigned int> idxs;
    };

    Mesh mesh;

    static inline int NUM_ITERS = 20;
    void do_thing() const;

    [[nodiscard]]
    static inline auto reset_mesh(Cone* cone) {
        // constexpr std::size_t NUM_ITERS = 20;
        const float ITER_STEP =
            glm::two_pi<float>() / static_cast<float>(NUM_ITERS);
        auto verts = std::vector<glm::vec3> {};
        verts.reserve(NUM_ITERS + 2);

        const auto& start = cone->start_position;
        const auto& dir_vec = cone->vec_norm;
        const auto lens = cone->vec_len / glm::cos(cone->angle_rads);
        const auto perp = glm::cross(cone->vec_norm, glm::vec3 {-1, -1, -1});
        const auto right_vec =
            lens * glm::rotate(cone->vec_norm, cone->angle_rads, perp);

        verts.emplace_back(start);

        float ang = 0.0f;
        while (ang < glm::two_pi<float>()) {
            // rotate around x axis
            const auto rotated = glm::rotate(right_vec, ang, dir_vec);
            verts.push_back(start + (rotated / 2.0f));
            verts.push_back(start + rotated);
            ang += ITER_STEP;
        }

        auto idxs = std::vector<unsigned int> {};
        idxs.reserve(NUM_ITERS * 9);
        {
            auto i = 0;
            for (int iter = 0; iter < NUM_ITERS - 1; ++iter) {
                idxs.push_back(0);
                idxs.push_back(i + 1);
                idxs.push_back(i + 3);

                idxs.push_back(i + 1);
                idxs.push_back(i + 2);
                idxs.push_back(i + 4);

                idxs.push_back(i + 1);
                idxs.push_back(i + 4);
                idxs.push_back(i + 3);
                i += 2;
            }
        }
        auto v_len = static_cast<unsigned int>(verts.size());
        idxs.push_back(0);
        idxs.push_back(v_len - 2);
        idxs.push_back(1);

        idxs.push_back(v_len - 2);
        idxs.push_back(v_len - 1);
        idxs.push_back(2);

        idxs.push_back(v_len - 2);
        idxs.push_back(2);
        idxs.push_back(1);

        return Mesh {verts, idxs};
    }

    inline void
    reset(const glm::vec3& start_point, const ::glm::vec3& end_point) {
        this->start_position = start_point;
        this->end_position = end_point;

        const auto v = end_point - start_point;
        this->raw_vector = v;
        this->vec_norm = glm::normalize(v);
        this->vec_len = glm::length(v);
        this->mesh = reset_mesh(this);
    }

    Cone() = default;

    Cone(const glm::vec3& start_point, const glm::vec3& end_point) {
        this->reset(start_point, end_point);
    }

    static constexpr const char* SERIALIZE_FMT_STR =
        "vec3(%f,%f,%f),vec3(%f,%f,%f),%f";

    inline std::string serialize() const {
        std::stringstream s {};
        s << glm::to_string(this->start_position) << ',';
        s << glm::to_string(this->end_position) << ',';
        s << this->angle_rads;
        return s.str();
    }

    /*
     * Finds the best frustum/cone angle from:
     *  - max velocity of any obj
     *  - runtime for the object to move
     *  - distance for the ship to travel
     *
     *  Look at max asteroid velocity and runtime, then use that to create frustum
     *  frustum_ang = atan[ (max vel * avg_runtime) / run_length ]
     */
    static inline float
    calculate_angle(float max_vel, float runtime_secs, float run_length) {
        return std::atan((max_vel * runtime_secs) / run_length);
    }

    inline void set_angle_degrees(float ang_degrees) {
        this->angle_rads = glm::radians(ang_degrees);
        this->mesh = reset_mesh(this);
    }

    /*
     * Sets a new start_position of the Cone and recalculates all the supporting values
     */
    inline void update_start_pos(const glm::vec3& new_pos) {
        this->reset(new_pos, this->end_position);
    }

    /*
     * Finds whether the point lays within this cone.
     *  This occurs when:
     *      - dot product > 0.0
     *      - ang_between cone vector and point vector is < half cone angle
     */
    inline bool check_collision(const glm::vec3& point) const {
        const auto point_vec = point - start_position;
        const auto dot = glm::dot(point_vec, this->raw_vector);
        const auto ang =
            glm::acos(dot / (glm::length(point_vec) * this->vec_len));
        return dot > 0.0f && ang < this->angle_rads;
    }

    /*
     * For man points finds whether the point lays within this cone.
     *  This occurs when:
     *      - dot product > 0.0
     *      - ang_between cone vector and point vector is < half cone angle
     */
    inline auto check_collision_many(std::span<const glm::vec3> points) const {
        return points | std::views::transform([&](const auto& p) {
                   return this->check_collision(p);
               });
    }

    /*
     * Gives the 3 points for the bounds of the cone, (about the Z axis)
     */
    inline std::tuple<glm::vec3, glm::vec3, glm::vec3> get_bounds() const {
        const auto right_norm = glm::rotateZ(this->vec_norm, this->angle_rads);
        const auto left_norm = glm::rotateZ(this->vec_norm, -this->angle_rads);

        const auto& start = this->start_position;
        const auto lens = this->vec_len / glm::cos(this->angle_rads);
        const auto right = start + (right_norm * lens);
        const auto left = start + (left_norm * lens);

        return std::tuple {start, right, left};
    }
};

struct Mask {
    using index_type = std::uint32_t;
    static constexpr std::size_t RESOLUTION = 1000;
    static constexpr std::size_t HALF_RES = RESOLUTION / 2;
    static constexpr float RESOLUTION_F = static_cast<float>(RESOLUTION);
    static constexpr float HALF_RES_F = static_cast<float>(HALF_RES);
    static_assert(
        RESOLUTION * RESOLUTION < std::numeric_limits<index_type>::max()
    );
    nd_array<index_type, RESOLUTION> mask;

    Mask() = default;

    inline void draw_object_over(
        const glm::vec3& ship_pos,
        const glm::vec3& target_pos,
        const glm::vec3& obj_pos,
        float radius
    ) {
        auto dir_vec = target_pos - ship_pos;
        auto obj_vec = obj_pos - ship_pos;
        auto dist_vec = obj_vec - dir_vec;
        // TODO:
        // Translate the positions onto the ships frame
        auto ang = vec_angle(dir_vec, obj_vec);
        glm::vec2 scaled {dist_vec.x, dist_vec.y};

        // Descretize / normalize to dimensions
        // Check for bounds
        if (glm::abs(scaled.x) > HALF_RES_F
            || (glm::abs(scaled.y) > HALF_RES_F))
            return;

        // Scale for distance away
        int radius_desc = static_cast<int>(radius / obj_vec.z);
        if (radius_desc == 0)
            return;

        const int desc_x = static_cast<int>(scaled.x);
        const int desc_y = static_cast<int>(scaled.y);
        const int y_min = std::max(0, desc_y - radius_desc);
        const int y_max =
            std::min(this->mask.num_rows<int>() - 1, desc_y + radius_desc);
        const int x_min = std::max(0, desc_x - radius_desc);
        const int x_max =
            std::min(this->mask.num_cols<int>() - 1, desc_x + radius_desc);
        const int r2 = radius_desc * radius_desc;

        // draw circle
        auto& grid = this->mask;
        for (int y = y_min; y <= y_max; ++y) {
            const int dy = y - desc_y;
            for (int x = x_min; x <= x_max; ++x) {
                const int dx = x - desc_x;
                if (dx * dx + dy * dy <= r2)
                    grid[y][x]++;
            }
        }
    }

    inline void print() const {
        auto fmt = std::format("{}", this->mask);
        printf("%s\n", fmt.c_str());
    }

    constexpr static inline Mask init_thing() {
        auto mask = collision::Mask {};
        auto f = std::array<unsigned int, mask.RESOLUTION>();
        f.fill(69);
        mask.mask.row(3) = f;
        return mask;
    }

    static inline auto& get_thing() {
        static auto thing = init_thing();
        return thing.mask;
    }
};

} // namespace collision
