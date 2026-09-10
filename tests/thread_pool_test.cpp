#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <thread>
#include <vector>

#include "ash/io/thread_pool.hpp"
#include "ash/task.hpp"

namespace {

ash::Task<int> doubled_on_pool(ash::ThreadPool& pool, int value) {
    co_await ash::schedule_on(pool);
    co_return value * 2;
}

}  // namespace

TEST_CASE("thread pool runs every posted task", "[thread_pool]") {
    ash::ThreadPool pool{4};

    std::atomic<int> counter{0};
    for (int i = 0; i < 200; ++i) {
        pool.post([&counter] { counter.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.wait_idle();
    REQUIRE(counter.load() == 200);
}

TEST_CASE("thread pool reports its size", "[thread_pool]") {
    ash::ThreadPool pool{3};
    REQUIRE(pool.size() == 3);
}

TEST_CASE("posted work runs on a worker thread", "[thread_pool]") {
    ash::ThreadPool pool{2};

    std::atomic<bool> ran_elsewhere{false};
    const auto caller = std::this_thread::get_id();
    pool.post([&ran_elsewhere, caller] {
        ran_elsewhere.store(std::this_thread::get_id() != caller, std::memory_order_relaxed);
    });

    pool.wait_idle();
    REQUIRE(ran_elsewhere.load());
}

TEST_CASE("a coroutine resumes on the pool after scheduling", "[thread_pool]") {
    ash::ThreadPool pool{2};
    REQUIRE(doubled_on_pool(pool, 21).sync_wait() == 42);
}

TEST_CASE("many coroutines share the pool concurrently", "[thread_pool]") {
    ash::ThreadPool pool{4};

    std::vector<ash::Task<int>> tasks;
    tasks.reserve(64);
    for (int i = 0; i < 64; ++i) {
        tasks.push_back(doubled_on_pool(pool, i));
    }

    int total = 0;
    for (auto& task : tasks) {
        total += task.sync_wait();
    }

    REQUIRE(total == 2 * (63 * 64 / 2));
}
