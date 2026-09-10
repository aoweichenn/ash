#include "ash/io/thread_pool.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace ash {

struct ThreadPool::Impl {
    std::mutex mutex;
    std::condition_variable_any work_available;
    std::condition_variable_any idle;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;
    std::stop_source stop;
    std::size_t active{0};
    std::size_t thread_count{0};

    explicit Impl(std::size_t count) : thread_count(count) {
        workers.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            workers.emplace_back([this] { run(); });
        }
    }

    ~Impl() {
        stop.request_stop();
        work_available.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void run() {
        const std::stop_token token = stop.get_token();
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock{mutex};
                work_available.wait(lock, token, [this] { return !queue.empty() || stop.stop_requested(); });
                if (queue.empty()) {
                    return;  // stopped and drained
                }
                task = std::move(queue.front());
                queue.pop_front();
                ++active;
            }

            task();

            {
                std::lock_guard lock{mutex};
                --active;
                if (queue.empty() && active == 0) {
                    idle.notify_all();
                }
            }
        }
    }

    void post(std::function<void()> fn) {
        {
            std::lock_guard lock{mutex};
            queue.push_back(std::move(fn));
        }
        work_available.notify_one();
    }

    void wait_idle() {
        std::unique_lock lock{mutex};
        idle.wait(lock, [this] { return queue.empty() && active == 0; });
    }
};

ThreadPool::ThreadPool(std::size_t thread_count)
    : impl_(std::make_unique<Impl>(thread_count == 0 ? std::max<std::size_t>(1, std::thread::hardware_concurrency())
                                                     : thread_count)) {}

ThreadPool::~ThreadPool() = default;

void ThreadPool::post(std::function<void()> fn) { impl_->post(std::move(fn)); }

void ThreadPool::wait_idle() { impl_->wait_idle(); }

std::size_t ThreadPool::size() const noexcept { return impl_->thread_count; }

}  // namespace ash
