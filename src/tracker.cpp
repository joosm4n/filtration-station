#include "tracker.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>

#include "parallel.hpp"

namespace {

std::unique_ptr<FilterBase> make(const TrackerConfig& c, double x0) {
    if (c.filter == FilterType::KALMAN) {
        return std::make_unique<KalmanFilter>(
            x0,
            c.v0,
            0.0,
            c.q,
            std::max(c.sensor.sig_pos, 1e-4),
            c.p0
        );
    }
    return std::make_unique<AbgFilter>(x0, c.v0, 0.0, c.alpha, c.beta, c.gamma);
}

} // namespace

// ---- Tracker --------------------------------------------------------------------------

bool Tracker::update(const Measurement& m) {
    if (!std::isfinite(m.t) || !std::isfinite(m.x) || !std::isfinite(m.y)
        || !std::isfinite(m.z))
        return false;

    if (!fx_) { // first report: start the filters on it
        fx_ = make(cfg_, m.x);
        fy_ = make(cfg_, m.y);
        fz_ = make(cfg_, m.z);
    } else {
        const double dt = m.t - t_last_;
        if (dt <= 0.0)
            return false;
        fx_->predict(dt);
        fy_->predict(dt);
        fz_->predict(dt);
        fx_->update(dt, fx_->innovation(m.x));
        fy_->update(dt, fy_->innovation(m.y));
        fz_->update(dt, fz_->innovation(m.z));
    }
    t_last_ = m.t;
    ++n_;

    if (std::isfinite(m.d) && m.d > 0.0) {
        d_sum_ += m.d;
        ++d_n_;
    }
    // TODO: rc and vr are accepted but not fused yet (see notes: needs an EKF update).
    return true;
}

Prediction Tracker::predict(double t) const {
    if (!fx_)
        throw std::logic_error("Tracker::predict: no measurements yet");
    const double h = t - t_last_;
    if (!(h >= -1e-12))
        throw std::invalid_argument(
            "Tracker::predict: time is before the last measurement"
        );

    const double hh = std::max(h, 0.0);
    const Forecast a = fx_->forecast(hh), b = fy_->forecast(hh),
                   c = fz_->forecast(hh);
    Prediction p;
    p.t = t;
    p.x = a.x;
    p.y = b.x;
    p.z = c.x;
    p.vx = a.v;
    p.vy = b.v;
    p.vz = c.v;
    p.sx = std::sqrt(a.var_x); // NaN for ABG
    p.sy = std::sqrt(b.var_x);
    p.sz = std::sqrt(c.var_x);
    return p;
}

// ---- TrackerBank ----------------------------------------------------------------------

int TrackerBank::ingest(const std::vector<Measurement>& ms) {
    std::unique_lock lock(mx_);

    std::map<int64_t, std::vector<const Measurement*>> groups;
    for (const Measurement& m : ms)
        groups[m.id].push_back(&m);

    struct Job {
        Tracker* tr;
        std::vector<const Measurement*>* list;
    };

    std::vector<Job> jobs;
    jobs.reserve(groups.size());
    for (auto& [id, list] : groups) {
        std::stable_sort(list.begin(), list.end(), [](auto* a, auto* b) {
            return a->t < b->t;
        });
        Tracker& tr = trackers_.try_emplace(id, cfg_).first->second;
        jobs.push_back({&tr, &list});
    }

    // each Tracker is touched by exactly one thread
    std::atomic<int> rejected {0};
    parallel_for(static_cast<int>(jobs.size()), threads_, [&](int i) {
        int bad = 0;
        for (const Measurement* m : *jobs[i].list)
            if (!jobs[i].tr->update(*m))
                ++bad;
        rejected += bad;
    });
    return rejected;
}

Prediction TrackerBank::predict(int64_t id, double t) const {
    std::shared_lock lock(mx_);
    const auto it = trackers_.find(id);
    if (it == trackers_.end())
        throw std::out_of_range("TrackerBank::predict: unknown id");
    Prediction p = it->second.predict(t);
    p.id = id;
    return p;
}

std::vector<Prediction> TrackerBank::predict_all(double t) const {
    std::shared_lock lock(mx_);

    std::vector<const std::pair<const int64_t, Tracker>*> items;
    for (const auto& kv : trackers_) {
        if (!kv.second.initialised())
            continue;
        if (!(t - kv.second.last_time()
              >= -1e-12)) // validate before going parallel
            throw std::invalid_argument(
                "TrackerBank::predict_all: time is before a measurement"
            );
        items.push_back(&kv);
    }

    std::vector<Prediction> out(items.size());
    parallel_for(static_cast<int>(items.size()), threads_, [&](int i) {
        out[i] = items[i]->second.predict(t);
        out[i].id = items[i]->first;
    });
    return out;
}

std::vector<int64_t> TrackerBank::ids() const {
    std::shared_lock lock(mx_);
    std::vector<int64_t> v;
    for (const auto& kv : trackers_)
        v.push_back(kv.first);
    return v;
}

double TrackerBank::diameter(int64_t id) const {
    std::shared_lock lock(mx_);
    const auto it = trackers_.find(id);
    if (it == trackers_.end())
        throw std::out_of_range("TrackerBank::diameter: unknown id");
    return it->second.diameter();
}

void TrackerBank::reset() {
    std::unique_lock lock(mx_);
    trackers_.clear();
}
