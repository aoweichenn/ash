#pragma once

#include <coroutine>
#include <cstddef>
#include <deque>
#include <mutex>

namespace ash {

// A counting semaphore whose waiters suspend their coroutine instead of
// blocking their thread.
//
// The distinction is the whole point. A std::counting_semaphore acquired inside
// a coroutine parks the worker thread until a permit frees up, so a pool of N
// threads can be entirely occupied by tasks waiting on permits that only tasks
// in that same pool could release. It deadlocks, and it does so under exactly
// the load the limiter was added to handle. Here a task that cannot get a
// permit gives up its thread and is resumed by whoever releases one.
//
// The cost of that: a parked coroutine is remembered by raw handle, so it must
// outlive its wait. Destroying a Task while it sits in this queue would leave
// release() resuming freed memory, and there is no way to tell from here that
// it happened. Every user in this runtime holds its tasks until they finish,
// which is what makes that safe; a cancellation path that drops parked work
// would need a registration to unlink first.
class AsyncSemaphore {
public:
    // Zero permits means nobody ever gets in, which is never what a caller
    // means -- so a limit of zero has to be spelled as "no limiter at all" by
    // the caller rather than passed here.
    explicit AsyncSemaphore(std::size_t permits) noexcept : permits_(permits) {}

    AsyncSemaphore(const AsyncSemaphore&) = delete;
    AsyncSemaphore& operator=(const AsyncSemaphore&) = delete;

    class AcquireAwaiter {
    public:
        explicit AcquireAwaiter(AsyncSemaphore& semaphore) noexcept : semaphore_(&semaphore) {}

        bool await_ready() const noexcept { return false; }

        // False resumes the coroutine straight away; true leaves it parked
        // until release() hands it a permit. Deciding here rather than in
        // await_ready keeps the check and the reservation one atomic step, so
        // two coroutines cannot both observe the last permit.
        bool await_suspend(std::coroutine_handle<> handle) { return semaphore_->acquire_or_park(handle); }

        void await_resume() const noexcept {}

    private:
        AsyncSemaphore* semaphore_;
    };

    AcquireAwaiter acquire() noexcept { return AcquireAwaiter{*this}; }

    // Hands the permit to the longest-waiting coroutine, or returns it to the
    // pool when nobody is waiting. Resuming happens outside the lock, because
    // the resumed coroutine may acquire again immediately.
    void release() noexcept {
        std::coroutine_handle<> next{};
        {
            const std::lock_guard lock{mutex_};
            if (waiters_.empty()) {
                ++permits_;
                return;
            }
            next = waiters_.front();
            waiters_.pop_front();
        }
        next.resume();
    }

private:
    [[nodiscard]] bool acquire_or_park(std::coroutine_handle<> handle) {
        const std::lock_guard lock{mutex_};
        if (permits_ > 0) {
            --permits_;
            return false;
        }
        waiters_.push_back(handle);
        return true;
    }

    mutable std::mutex mutex_;
    std::deque<std::coroutine_handle<>> waiters_;
    std::size_t permits_;
};

}  // namespace ash
