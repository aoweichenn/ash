#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "ash/io/thread_pool.hpp"
#include "ash/task.hpp"

namespace ash {

namespace detail {

// Counts completions from any number of threads.
//
// A std::binary_semaphore would be the obvious choice and is wrong here: its
// counter saturates at one, so if every worker finishes before the waiter
// starts acquiring, the extra releases are dropped on the floor and the waiter
// blocks forever. That is a race that shows up on a fast machine and never on a
// slow one, which is the worst way to find out.
class CompletionCounter {
public:
    void complete() {
        {
            const std::lock_guard lock{mutex_};
            ++completed_;
        }
        condition_.notify_all();
    }

    void wait_for(std::size_t count) {
        std::unique_lock lock{mutex_};
        condition_.wait(lock, [&] { return completed_ >= count; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t completed_ = 0;
};

// Runs a child to completion and records what it produced, so that when_all can
// collect a batch it never waited on directly.
//
// Started on a pool worker. When the child suspends -- on a provider limiter, on
// a hop back to the pool -- the worker returns to the queue instead of being
// held until the child finishes. That is the difference between a batch that can
// be larger than the pool and one that cannot: a batch of a hundred calls on
// twenty threads only works if a call waiting for a permit is not also
// occupying the thread it would need to be resumed on.
template <class T>
Task<void> run_child(Task<T> child, std::optional<T>& value, std::exception_ptr& error, CompletionCounter& done) {
    try {
        value.emplace(co_await std::move(child));
    } catch (...) {
        // Recorded rather than rethrown: throwing here would unwind through the
        // pool's worker and lose the exception.
        error = std::current_exception();
    }
    done.complete();
}

}  // namespace detail

// Runs every task on the pool and returns their results in the order the tasks
// were given, not the order they happened to finish. Blocks until all of them
// have completed, so a task can never outlive the call.
//
// The batch may be larger than the pool. Each task is started on a worker and
// gives that worker back the moment it suspends, so a hundred calls on twenty
// threads make progress twenty at a time rather than deadlocking at twenty.
// Only the calling thread is held for the duration, and holding it is the point
// -- this is the call that turns a fan-out back into a single answer.
//
// The first exception, in task order, is rethrown once every task has finished.
// That ordering matters: reporting whichever failure happened to land first
// would make the message depend on scheduling, and the point of this runtime is
// that the same input gives the same answer twice.
//
// Value tasks only. Fanning out work with no result is what Nursery is for, and
// constraining the template there says so in one place instead of making every
// caller reason about an empty result vector.
//
// Must not be called from a pool worker. The tasks are queued behind whatever
// is already running, so blocking a worker while waiting for work that worker
// would have to run is a deadlock.
template <class T>
    requires(!std::is_void_v<T>)
std::vector<T> when_all(ThreadPool& pool, std::vector<Task<T>> tasks) {
    const std::size_t count = tasks.size();

    std::vector<std::optional<T>> results(count);
    std::vector<std::exception_ptr> errors(count);

    detail::CompletionCounter completed;

    // The runners are themselves coroutines, so something has to own them until
    // they finish. These do, and they outlive the wait below -- which is also
    // what keeps the frames each runner holds its child in alive.
    std::vector<std::shared_ptr<Task<void>>> runners;
    runners.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        // post() takes a std::function, which has to be copyable, and a Task is
        // move-only -- so each runner is owned through a shared_ptr rather than
        // captured by value.
        auto runner = std::make_shared<Task<void>>(
            detail::run_child(std::move(tasks[i]), results[i], errors[i], completed));
        runners.push_back(runner);
        pool.post([runner] { runner->start(); });
    }

    completed.wait_for(count);

    for (const std::exception_ptr& error : errors) {
        if (error != nullptr) {
            std::rethrow_exception(error);
        }
    }

    std::vector<T> ordered;
    ordered.reserve(count);
    for (std::optional<T>& slot : results) {
        ordered.push_back(std::move(*slot));
    }
    return ordered;
}

// Structured concurrency with a shared stop token.
//
// The nursery owns the work it spawns. It cannot be destroyed while a child is
// still running, wait() does not return until every child has finished, and the
// first child to fail asks the others to stop rather than letting them run on
// with nobody watching. Children read stop_token() and pass it down -- in this
// runtime that reaches the agent loop and its tools, so a cancelled run stops
// rather than being ignored.
class Nursery {
public:
    explicit Nursery(ThreadPool& pool) noexcept : pool_(&pool) {}

    Nursery(const Nursery&) = delete;
    Nursery& operator=(const Nursery&) = delete;

    // A nursery destroyed without wait() cancels what is left and then waits
    // for it, because a child holding a pointer to this object must not outlive
    // it. Nothing is detached.
    ~Nursery() { drain(); }

    [[nodiscard]] ThreadPool& pool() const noexcept { return *pool_; }

    [[nodiscard]] std::stop_token stop_token() const noexcept { return source_.get_token(); }

    // Queues a child on the pool. Spawning after wait() is a programming error
    // and throws, rather than running work nobody will ever look at.
    void spawn(std::function<void()> body) {
        {
            const std::lock_guard lock{mutex_};
            if (waited_) {
                throw std::logic_error{"Nursery::spawn called after wait()"};
            }
            ++pending_;
        }

        pool_->post([this, body = std::move(body)]() mutable {
            try {
                body();
            } catch (...) {
                const std::lock_guard lock{mutex_};
                if (first_error_ == nullptr) {
                    first_error_ = std::current_exception();
                    // The siblings get to find out. Whether they act on it is
                    // their business, but nothing else is going to tell them.
                    source_.request_stop();
                }
            }
            completed_.complete();
        });
    }

    // Waits for every child. If one failed, its exception is rethrown -- after
    // the rest have finished, so no work is left running behind the throw.
    void wait() {
        const std::size_t pending = begin_wait();

        completed_.wait_for(pending);

        const std::lock_guard lock{mutex_};
        if (first_error_ != nullptr) {
            std::rethrow_exception(first_error_);
        }
    }

private:
    // Marks the nursery as waited and reports how many completions to expect.
    [[nodiscard]] std::size_t begin_wait() {
        const std::lock_guard lock{mutex_};
        if (waited_) {
            throw std::logic_error{"Nursery::wait called twice"};
        }
        waited_ = true;
        return pending_;
    }

    // Waits without throwing, for the destructor.
    void drain() noexcept {
        std::size_t pending = 0;
        {
            const std::lock_guard lock{mutex_};
            if (waited_) {
                return;
            }
            waited_ = true;
            pending = pending_;
        }
        completed_.wait_for(pending);
    }

    ThreadPool* pool_;
    std::stop_source source_;
    detail::CompletionCounter completed_;
    std::mutex mutex_;
    std::exception_ptr first_error_{};
    std::size_t pending_ = 0;
    bool waited_ = false;
};

}  // namespace ash
