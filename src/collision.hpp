#pragma once

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

    [[nodiscard]]
    static inline auto reset_mesh(Cone* cone) {
        // constexpr std::size_t NUM_ITERS = 20;V
        float ITER_STEP = glm::two_pi<float>() / NUM_ITERS;
        auto verts = std::vector<glm::vec3> {};
        verts.reserve(NUM_ITERS + 1);

        const auto& start = cone->start_position;
        const auto& dir_vec = cone->vec_norm;
        const auto lens = cone->vec_len / glm::cos(cone->angle_rads);
        const auto perp = glm::cross(cone->vec_norm, glm::vec3 {-1, -1, -1});
        const auto right_norm =
            glm::rotate(cone->vec_norm, cone->angle_rads, perp) * lens;
        const auto right = start + right_norm;

        verts.emplace_back(start);
        verts.emplace_back(right);
        // printf(
        //     "right: %s, right_norm: %s, lens: %f\n",
        //     glm::to_string(right).c_str(),
        //     glm::to_string(right_norm).c_str(),
        //     lens
        // );

        float ang = 0.0f;
        for (std::size_t i = 0; i <= NUM_ITERS; ++i) {
            // rotate around x axis
            const auto rotated = glm::rotate(right_norm, ang, dir_vec);
            // printf("rotated: %s\n", glm::to_string(rotated).c_str());
            const auto shifted = start + rotated;
            // printf("shifted: %s\n", glm::to_string(rotated).c_str());
            verts.push_back(shifted);
            ang += ITER_STEP;
        }

        auto idxs = std::vector<unsigned int> {};
        idxs.reserve(NUM_ITERS * 3);
        for (unsigned int i = 0; i <= NUM_ITERS; ++i) {
            idxs.push_back(0);
            idxs.push_back(i + 1);
            idxs.push_back(i + 2);
        }

        // printf("verts: \n");
        // for (const auto& v : verts) {
        //     printf("%s\n", glm::to_string(v).c_str());
        // }
        // printf("\n");

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

} // namespace collision
