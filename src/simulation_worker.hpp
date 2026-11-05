#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

template<class Params, class Result, Result (*Run)(const Params&)>
class SimulationWorker {
  public:
    SimulationWorker() : thread_([this] { loop(); }) {}
    ~SimulationWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            quit_ = true;
        }
        condition_.notify_one();
        thread_.join();
    }

    SimulationWorker(const SimulationWorker&) = delete;
    SimulationWorker& operator=(const SimulationWorker&) = delete;

    void request(const Params& params) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_ = params;
        }
        condition_.notify_one();
    }

    std::optional<Result> take() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::optional<Result> result;
        result.swap(ready_);
        return result;
    }

    bool busy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return running_ || pending_.has_value();
    }

  private:
    void loop() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            condition_.wait(lock, [this] { return quit_ || pending_.has_value(); });
            if (quit_)
                return;
            const Params params = *pending_;
            pending_.reset();
            running_ = true;
            lock.unlock();
            Result result = Run(params);
            lock.lock();
            running_ = false;
            ready_ = std::move(result);
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<Params> pending_;
    std::optional<Result> ready_;
    bool running_ = false;
    bool quit_ = false;
    std::thread thread_;
};
