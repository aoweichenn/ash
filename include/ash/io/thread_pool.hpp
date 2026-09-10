#pragma once

#include <coroutine>
#include <cstddef>
#include <functional>
#include <memory>

namespace ash {

// A fixed-size pool of worker threads draining a shared FIFO queue.
class ThreadPool {
public:
    // Zero means "one thread per hardware core".
    explicit ThreadPool(std::size_t thread_count = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void post(std::function<void()> fn);

    // Blocks until the queue is empty and no worker is mid-task.
    void wait_idle();

    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Hands the current coroutine to `pool` and resumes it on a worker thread.
class ScheduleAwaiter {
public:
    explicit ScheduleAwaiter(ThreadPool& pool) noexcept : pool_(&pool) {}

    bool await_ready() const noexcept { return false; }

    void await_suspend(std::coroutine_handle<> handle) const {
        pool_->post([handle] { handle.resume(); });
    }

    void await_resume() const noexcept {}

private:
    ThreadPool* pool_;
};

[[nodiscard]] inline ScheduleAwaiter schedule_on(ThreadPool& pool) noexcept {
    return ScheduleAwaiter{pool};
}

}  // namespace ash
