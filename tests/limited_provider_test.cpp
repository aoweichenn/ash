#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ash/concurrency.hpp"
#include "ash/io/thread_pool.hpp"
#include "ash/model/limited_provider.hpp"
#include "ash/model/provider.hpp"
#include "ash/task.hpp"

namespace {

using namespace std::chrono_literals;

// Records the largest number of calls that were inside chat at the same time.
//
// When `expected_peak` calls have arrived it releases them all, so a peak is
// reached because the limiter allowed it rather than because the machine
// happened to schedule those calls together.
//
// The rendezvous only decides anything when the pool is exactly `expected_peak`
// wide: then every call is running and none can leave until they have all
// arrived, so the count is forced. Give it more workers than that and the
// arrivals interleave with the departures, no call necessarily observes the
// full width, and the wait below just burns its timeout -- which is why the
// tests that use it pair it with a pool of that size.
class CountingProvider : public ash::ModelProvider {
public:
    CountingProvider(int expected_peak, ash::ThreadPool* park_on) : expected_peak_(expected_peak), park_on_(park_on) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "counting"; }
    [[nodiscard]] const std::string& model() const noexcept override { return model_; }

    ash::Task<ash::ChatResponse> chat(ash::ChatRequest) override {
        const int now = in_flight_.fetch_add(1) + 1;
        int seen = peak_.load();
        while (seen < now && !peak_.compare_exchange_weak(seen, now)) {
        }

        {
            std::unique_lock lock{mutex_};
            ++arrivals_;
            if (arrivals_ >= expected_peak_) {
                arrived_.notify_all();
            } else {
                // Waits on a count that only ever rises. Waiting on in_flight
                // instead would lose the wakeup: the call that opens the gate
                // decrements on its way out, so by the time the others have the
                // lock back the value they were woken for is gone and they
                // sleep out the timeout. Bounded, so a limiter more restrictive
                // than the test expects fails here instead of hanging.
                arrived_.wait_for(lock, 10s, [&] { return arrivals_ >= expected_peak_; });
            }
        }

        if (park_on_ != nullptr) {
            // Hands the worker back to the pool and waits to be resumed. A
            // limiter that blocks its thread while waiting for a permit has
            // nothing left to make progress with, which is the deadlock
            // AsyncSemaphore exists to avoid.
            co_await ash::schedule_on(*park_on_);
        }

        in_flight_.fetch_sub(1);
        co_return ash::ChatResponse{};
    }

    [[nodiscard]] int peak() const noexcept { return peak_.load(); }

private:
    std::string model_ = "counting-model";
    int expected_peak_;
    ash::ThreadPool* park_on_;
    std::atomic<int> in_flight_{0};
    std::atomic<int> peak_{0};
    std::mutex mutex_;
    std::condition_variable arrived_;
    int arrivals_ = 0;
};

class FailingProvider : public ash::ModelProvider {
public:
    [[nodiscard]] std::string_view name() const noexcept override { return "failing"; }
    [[nodiscard]] const std::string& model() const noexcept override { return model_; }

    ash::Task<ash::ChatResponse> chat(ash::ChatRequest) override {
        co_await std::suspend_never{};
        throw std::runtime_error{"provider is down"};
    }

private:
    std::string model_ = "failing-model";
};

std::vector<ash::Task<ash::ChatResponse>> calls_to(ash::ModelProvider& provider, int count) {
    std::vector<ash::Task<ash::ChatResponse>> tasks;
    tasks.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        tasks.push_back(provider.chat(ash::ChatRequest{}));
    }
    return tasks;
}

}  // namespace

TEST_CASE("the limiter caps how many calls are in flight", "[limited]") {
    ash::ThreadPool pool{8};

    auto inner = std::make_unique<CountingProvider>(1, nullptr);
    CountingProvider& counting = *inner;
    ash::LimitedProvider provider{std::move(inner), 3};

    // Six calls with eight workers to run them on, so anything above three in
    // flight is the limiter failing rather than the pool being busy. The count
    // is read inside the call, which only the holder of a permit reaches, so
    // the bound holds on any schedule.
    REQUIRE_NOTHROW(ash::when_all(pool, calls_to(provider, 6)));
    CHECK(counting.peak() <= 3);
}

TEST_CASE("the limiter holds the line when there are workers to spare", "[limited]") {
    ash::ThreadPool pool{8};

    auto inner = std::make_unique<CountingProvider>(1, nullptr);
    CountingProvider& counting = *inner;
    ash::LimitedProvider provider{std::move(inner), 1};

    // Eight workers, one permit. Admitting a second call would show up as a
    // peak of two, so an exact peak of one says the limit is what bound the
    // calls and not the size of the pool.
    REQUIRE_NOTHROW(ash::when_all(pool, calls_to(provider, 6)));
    CHECK(counting.peak() == 1);
}

TEST_CASE("the limiter admits its full width", "[limited]") {
    constexpr int kLimit = 3;
    // Pool exactly as wide as the limit, so all three calls are running and
    // each waits for the others -- the count is forced to reach three rather
    // than merely allowed to.
    ash::ThreadPool pool{kLimit};

    auto inner = std::make_unique<CountingProvider>(kLimit, nullptr);
    CountingProvider& counting = *inner;
    ash::LimitedProvider provider{std::move(inner), kLimit};

    REQUIRE_NOTHROW(ash::when_all(pool, calls_to(provider, kLimit)));
    CHECK(counting.peak() == kLimit);
}

TEST_CASE("waiting for a permit does not hold a worker thread", "[limited]") {
    // One worker for three calls, each of which hands the worker back while it
    // waits. A limiter that parked the thread instead would leave nobody to
    // resume the call that holds the permit.
    ash::ThreadPool pool{1};

    auto inner = std::make_unique<CountingProvider>(1, &pool);
    ash::LimitedProvider provider{std::move(inner), 1};

    REQUIRE_NOTHROW(ash::when_all(pool, calls_to(provider, 3)));
}

TEST_CASE("a failed call gives its permit back", "[limited]") {
    ash::ThreadPool pool{4};

    auto inner = std::make_unique<FailingProvider>();
    ash::LimitedProvider provider{std::move(inner), 1};

    // Four calls through a single permit. A permit leaked on the throw would
    // leave the second call waiting for a slot that never comes back, and the
    // whole batch would hang rather than fail.
    REQUIRE_THROWS_AS(ash::when_all(pool, calls_to(provider, 4)), std::runtime_error);
}

// A limit of zero is not a limit, it is a provider that can never be called,
// and it would show up as a hang rather than as a mistake.
TEST_CASE("a limiter with no permits is refused", "[limited]") {
    auto inner = std::make_unique<CountingProvider>(1, nullptr);
    // Parenthesised: the comma inside the braces would otherwise read as a
    // second argument to the assertion macro.
    CHECK_THROWS_AS((ash::LimitedProvider{std::move(inner), 0}), std::invalid_argument);
}

TEST_CASE("a limiter with nothing to wrap is refused", "[limited]") {
    CHECK_THROWS_AS((ash::LimitedProvider{nullptr, 4}), std::invalid_argument);
}

TEST_CASE("a limiter passes through what it wraps", "[limited]") {
    auto inner = std::make_unique<CountingProvider>(1, nullptr);
    ash::LimitedProvider provider{std::move(inner), 4};

    CHECK(provider.name() == "counting");
    CHECK(provider.model() == "counting-model");
    CHECK(provider.max_inflight() == 4);
}
