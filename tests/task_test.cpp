#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <coroutine>
#include <stdexcept>
#include <string>
#include <utility>

#include "ash/io/thread_pool.hpp"
#include "ash/task.hpp"

namespace {

ash::Task<int> answer() { co_return 42; }

ash::Task<std::string> greet(std::string name) { co_return "hello " + name; }

ash::Task<int> sum_of_two_awaits() {
    const int a = co_await answer();
    const int b = co_await answer();
    co_return a + b;
}

ash::Task<int> throws_before_returning() {
    // Without a co_ keyword this would be an ordinary function: the throw would
    // unwind at the call site and the promise's unhandled_exception would never
    // run, so the test would pass without touching the code it names.
    co_await std::suspend_never{};
    throw std::runtime_error{"boom"};
}

ash::Task<void> completes_void() { co_return; }

}  // namespace

TEST_CASE("task yields its value through sync_wait", "[task]") {
    REQUIRE(answer().sync_wait() == 42);
}

TEST_CASE("task arguments survive into the coroutine frame", "[task]") {
    REQUIRE(greet("ash").sync_wait() == "hello ash");
}

TEST_CASE("tasks compose through co_await", "[task]") {
    REQUIRE(sum_of_two_awaits().sync_wait() == 84);
}

TEST_CASE("exceptions propagate to the awaiting caller", "[task]") {
    REQUIRE_THROWS_AS(throws_before_returning().sync_wait(), std::runtime_error);
}

TEST_CASE("void tasks run to completion", "[task]") {
    completes_void().sync_wait();
    SUCCEED("void task completed");
}

TEST_CASE("a started task runs without the caller waiting on it", "[task]") {
    ash::ThreadPool pool{1};
    std::atomic<bool> finished{false};

    auto hop = [&]() -> ash::Task<int> {
        co_await ash::schedule_on(pool);
        finished.store(true);
        co_return 7;
    };

    ash::Task<int> task = hop();
    task.start();
    // start() returns with the coroutine parked on the pool, so this is what
    // proves the rest of it ran: the frame has to stay alive until it does,
    // which is the obligation start() puts on its caller.
    pool.wait_idle();
    CHECK(finished.load());
}

TEST_CASE("a task is movable but not copyable", "[task]") {
    static_assert(std::is_move_constructible_v<ash::Task<int>>);
    static_assert(!std::is_copy_constructible_v<ash::Task<int>>);

    auto task = answer();
    auto moved = std::move(task);
    REQUIRE_FALSE(task.valid());
    REQUIRE(moved.sync_wait() == 42);
}
