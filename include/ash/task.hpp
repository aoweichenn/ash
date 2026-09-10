#pragma once

#include <coroutine>
#include <exception>
#include <optional>
#include <semaphore>
#include <type_traits>
#include <utility>

namespace ash {

namespace detail {

template <class T>
struct ResultStorage {
    std::optional<T> value{};
    std::exception_ptr error{};

    template <class U>
    void return_value(U&& v) {
        value.emplace(std::forward<U>(v));
    }

    T take() { return std::move(*value); }
};

template <>
struct ResultStorage<void> {
    std::exception_ptr error{};

    void return_void() noexcept {}

    void take() noexcept {}
};

// Transfers control to the awaiting coroutine on completion. Releasing the
// completion semaphore before handing off lets a blocking sync_wait() wake up
// even when the tail of the chain runs on a pool thread.
struct FinalAwaiter {
    bool await_ready() const noexcept { return false; }

    template <class Promise>
    std::coroutine_handle<> await_suspend(std::coroutine_handle<Promise> handle) noexcept {
        auto& promise = handle.promise();
        if (promise.completion != nullptr) {
            promise.completion->release();
        }
        return promise.continuation != nullptr ? promise.continuation : std::noop_coroutine();
    }

    void await_resume() const noexcept {}
};

}  // namespace detail

// A lazy, single-shot coroutine. Awaiting it transfers control to the child
// coroutine rather than calling into it, so long co_await chains run in
// constant stack space.
template <class T = void>
class Task {
public:
    struct promise_type : detail::ResultStorage<T> {
        std::coroutine_handle<> continuation{};
        std::binary_semaphore* completion{nullptr};

        Task get_return_object() noexcept {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        detail::FinalAwaiter final_suspend() noexcept { return {}; }

        void unhandled_exception() noexcept { this->error = std::current_exception(); }
    };

    Task() noexcept = default;

    explicit Task(std::coroutine_handle<promise_type> handle) noexcept : handle_(handle) {}

    Task(Task&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_ != nullptr) {
                handle_.destroy();
            }
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    ~Task() {
        if (handle_ != nullptr) {
            handle_.destroy();
        }
    }

    [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }

    class Awaiter {
    public:
        explicit Awaiter(std::coroutine_handle<promise_type> handle) noexcept : handle_(handle) {}

        bool await_ready() const noexcept { return false; }

        std::coroutine_handle<> await_suspend(std::coroutine_handle<> continuation) const noexcept {
            handle_.promise().continuation = continuation;
            return handle_;
        }

        decltype(auto) await_resume() {
            auto& promise = handle_.promise();
            if (promise.error != nullptr) {
                std::rethrow_exception(promise.error);
            }
            if constexpr (!std::is_void_v<T>) {
                return promise.take();
            }
        }

    private:
        std::coroutine_handle<promise_type> handle_;
    };

    Awaiter operator co_await() && noexcept { return Awaiter{handle_}; }

    // Runs the coroutine to completion on the calling thread, blocking while it
    // is suspended (for example, parked on a thread pool).
    T sync_wait() {
        std::binary_semaphore done{0};
        handle_.promise().completion = &done;
        handle_.resume();
        done.acquire();
        handle_.promise().completion = nullptr;

        auto& promise = handle_.promise();
        if (promise.error != nullptr) {
            std::rethrow_exception(promise.error);
        }
        if constexpr (!std::is_void_v<T>) {
            return promise.take();
        }
    }

private:
    std::coroutine_handle<promise_type> handle_{};
};

}  // namespace ash
