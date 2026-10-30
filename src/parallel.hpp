#pragma once
// Minimal thread helper shared by the simulator and the tracker API.
#include <algorithm>
#include <atomic>
#include <functional>
#include <thread>
#include <vector>

// Run fn(0..n-1) on up to `max_threads` threads. The calling thread is one of them,
// so max_threads = 1 means no extra threads are created.
// fn must not throw (an exception escaping a worker thread terminates the program).
inline void
parallel_for(int n, int max_threads, const std::function<void(int)>& fn) {
    const int nthreads = std::clamp(max_threads, 1, std::max(n, 1));
    if (nthreads == 1) {
        for (int i = 0; i < n; ++i)
            fn(i);
        return;
    }
    std::atomic<int> next {0};
    auto work = [&] {
        for (int i; (i = next.fetch_add(1)) < n;)
            fn(i);
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (int i = 1; i < nthreads; ++i)
        pool.emplace_back(work);
    work();
    for (auto& th : pool)
        th.join();
}
