#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ash/concurrency.hpp"
#include "ash/io/thread_pool.hpp"
#include "ash/task.hpp"

namespace {

using namespace std::chrono_literals;

// A rendezvous point. Every task that reaches it waits for the rest, so a
// batch that ran one at a time would time out here rather than hang the suite
// forever -- the test fails, which is the point.
class Rendezvous {
public:
    explicit Rendezvous(std::size_t expected) : expected_(expected) {}

    bool arrive_and_wait() {
        std::unique_lock lock{mutex_};
        ++arrived_;
        condition_.notify_all();
        return condition_.wait_for(lock, 10s, [&] { return arrived_ >= expected_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t expected_;
    std::size_t arrived_ = 0;
};

ash::Task<int> delayed_value(int value, std::chrono::milliseconds delay) {
    std::this_thread::sleep_for(delay);
    co_return value;
}

ash::Task<int> failing(std::string message) {
    // A real suspension point, so this is a coroutine and the throw happens
    // inside it rather than at the call site.
    co_await std::suspend_never{};
    if (!message.empty()) {
        throw std::runtime_error{message};
    }
    co_return 0;
}

ash::Task<int> arriving(Rendezvous& rendezvous, int value) {
    if (!rendezvous.arrive_and_wait()) {
        throw std::runtime_error{"tasks did not overlap"};
    }
    co_return value;
}

}  // namespace

TEST_CASE("when_all returns results in task order, not completion order", "[concurrency]") {
    ash::ThreadPool pool{4};

    std::vector<ash::Task<int>> tasks;
    tasks.push_back(delayed_value(0, 80ms));
    tasks.push_back(delayed_value(1, 40ms));
    tasks.push_back(delayed_value(2, 0ms));
    tasks.push_back(delayed_value(3, 20ms));

    // They finish 2, 3, 1, 0.
    const std::vector<int> results = ash::when_all(pool, std::move(tasks));
    REQUIRE(results == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("when_all runs its tasks at the same time", "[concurrency]") {
    constexpr std::size_t kTasks = 4;
    ash::ThreadPool pool{kTasks};

    Rendezvous rendezvous{kTasks};
    std::vector<ash::Task<int>> tasks;
    for (std::size_t i = 0; i < kTasks; ++i) {
        tasks.push_back(arriving(rendezvous, static_cast<int>(i)));
    }

    // Each task blocks until every other one has arrived. Run sequentially
    // this cannot complete, so success here is proof of real overlap rather
    // than a timing measurement that could pass on a fast machine.
    REQUIRE(ash::when_all(pool, std::move(tasks)) == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("when_all rethrows the first failure in task order", "[concurrency]") {
    ash::ThreadPool pool{4};

    std::vector<ash::Task<int>> tasks;
    tasks.push_back(delayed_value(1, 60ms));
    tasks.push_back(failing("second"));
    tasks.push_back(failing("third"));
    tasks.push_back(delayed_value(4, 0ms));

    // "third" is likely to fail first, and "second" is the one that must be
    // reported: which failure surfaces cannot depend on the scheduler.
    try {
        ash::when_all(pool, std::move(tasks));
        FAIL("expected the batch to throw");
    } catch (const std::runtime_error& error) {
        REQUIRE(std::string{error.what()} == "second");
    }
}

TEST_CASE("when_all still waits for the other tasks before throwing", "[concurrency]") {
    ash::ThreadPool pool{4};

    std::atomic<bool> slow_finished{false};
    auto slow = [&]() -> ash::Task<int> {
        std::this_thread::sleep_for(60ms);
        slow_finished.store(true);
        co_return 1;
    };

    std::vector<ash::Task<int>> tasks;
    tasks.push_back(slow());
    tasks.push_back(failing("immediate"));

    REQUIRE_THROWS_AS(ash::when_all(pool, std::move(tasks)), std::runtime_error);
    // The failure must not leave work running behind it.
    REQUIRE(slow_finished.load());
}

TEST_CASE("when_all on an empty batch is a no-op", "[concurrency]") {
    ash::ThreadPool pool{2};

    const std::vector<int> results = ash::when_all(pool, std::vector<ash::Task<int>>{});
    REQUIRE(results.empty());
}

TEST_CASE("a nursery runs its children at the same time", "[concurrency]") {
    constexpr std::size_t kChildren = 4;
    ash::ThreadPool pool{kChildren};
    ash::Nursery nursery{pool};

    Rendezvous rendezvous{kChildren};
    for (std::size_t i = 0; i < kChildren; ++i) {
        nursery.spawn([&rendezvous] {
            // Same rendezvous as the when_all case: sequential children cannot
            // get past this, so no timing assumption is involved.
            if (!rendezvous.arrive_and_wait()) {
                throw std::runtime_error{"children did not overlap"};
            }
        });
    }

    REQUIRE_NOTHROW(nursery.wait());
}

TEST_CASE("a nursery runs every child before wait returns", "[concurrency]") {
    ash::ThreadPool pool{4};
    ash::Nursery nursery{pool};

    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        nursery.spawn([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }
    nursery.wait();

    REQUIRE(counter.load() == 50);
}

TEST_CASE("a nursery asks its siblings to stop when one fails", "[concurrency]") {
    ash::ThreadPool pool{4};
    ash::Nursery nursery{pool};

    std::atomic<bool> saw_stop{false};
    nursery.spawn([&nursery, &saw_stop] {
        // Bounded, so a nursery that never propagates the stop fails the test
        // instead of hanging it.
        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (std::chrono::steady_clock::now() < deadline) {
            if (nursery.stop_token().stop_requested()) {
                saw_stop.store(true);
                return;
            }
            std::this_thread::sleep_for(1ms);
        }
    });
    nursery.spawn([] { throw std::runtime_error{"child failed"}; });

    REQUIRE_THROWS_AS(nursery.wait(), std::runtime_error);
    REQUIRE(saw_stop.load());
}

TEST_CASE("a nursery reports the failure that happened", "[concurrency]") {
    ash::ThreadPool pool{2};
    ash::Nursery nursery{pool};

    nursery.spawn([] {});
    nursery.spawn([] { throw std::runtime_error{"the only failure"}; });

    try {
        nursery.wait();
        FAIL("expected the nursery to throw");
    } catch (const std::runtime_error& error) {
        REQUIRE(std::string{error.what()} == "the only failure");
    }
}

TEST_CASE("a nursery destroyed without wait finishes its children first", "[concurrency]") {
    ash::ThreadPool pool{2};
    std::atomic<bool> finished{false};

    {
        ash::Nursery nursery{pool};
        nursery.spawn([&finished] {
            std::this_thread::sleep_for(50ms);
            finished.store(true);
        });
        // Deliberately no wait(): the destructor has to drain, because a child
        // still holding a pointer to the nursery must not outlive it.
    }

    REQUIRE(finished.load());
}

TEST_CASE("a nursery refuses work it could not wait for", "[concurrency]") {
    ash::ThreadPool pool{2};
    ash::Nursery nursery{pool};

    nursery.spawn([] {});
    nursery.wait();

    REQUIRE_THROWS_AS(nursery.spawn([] {}), std::logic_error);
    REQUIRE_THROWS_AS(nursery.wait(), std::logic_error);
}

TEST_CASE("a nursery with no children waits for nothing", "[concurrency]") {
    ash::ThreadPool pool{2};
    ash::Nursery nursery{pool};

    REQUIRE_NOTHROW(nursery.wait());
    REQUIRE_FALSE(nursery.stop_token().stop_requested());
}
