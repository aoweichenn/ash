#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <csignal>
#include <stop_token>
#include <thread>

#include "terminal.hpp"

namespace {

// Swallows the signal so a test can raise one without the default action ending
// the process. Deliberately not ash::cli::InterruptHandler: a test that wants to
// check what the session's handler does has to be able to run without it.
class IgnoreInterrupt {
public:
    IgnoreInterrupt() {
        struct sigaction action {};
        action.sa_handler = SIG_IGN;
        action.sa_flags = 0;
        ::sigemptyset(&action.sa_mask);
        ::sigaction(SIGINT, &action, &previous_);
    }

    ~IgnoreInterrupt() { ::sigaction(SIGINT, &previous_, nullptr); }

    IgnoreInterrupt(const IgnoreInterrupt&) = delete;
    IgnoreInterrupt& operator=(const IgnoreInterrupt&) = delete;

private:
    struct sigaction previous_ {};
};

// Waits for something to become true rather than sleeping a guessed amount and
// hoping. Returns whether it did.
template <typename Predicate>
[[nodiscard]] bool wait_for(Predicate done, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (done()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return done();
}

}  // namespace

TEST_CASE("the handler records a Ctrl-C rather than acting on it") {
    // The point of the flag is that a handler cannot do anything else: stopping
    // a run takes a lock, and a handler that took one could deadlock against the
    // thread it interrupted.
    ash::cli::clear_interrupt();
    {
        const ash::cli::InterruptHandler handler;
        ::raise(SIGINT);
    }

    CHECK(ash::cli::g_interrupted != 0);
    ash::cli::clear_interrupt();
}

TEST_CASE("the handler is installed without SA_RESTART and put back afterwards") {
    // sa_flags == 0 is the load-bearing part and the easiest thing to lose. With
    // SA_RESTART the kernel restarts the read at the prompt for us, and the one
    // way the session has of noticing a Ctrl-C at an empty prompt disappears
    // without anything else visibly changing.
    struct sigaction before {};
    REQUIRE(::sigaction(SIGINT, nullptr, &before) == 0);

    {
        const ash::cli::InterruptHandler handler;
        struct sigaction during {};
        REQUIRE(::sigaction(SIGINT, nullptr, &during) == 0);

        CHECK(during.sa_handler != before.sa_handler);
        CHECK((during.sa_flags & SA_RESTART) == 0);
    }

    struct sigaction after {};
    REQUIRE(::sigaction(SIGINT, nullptr, &after) == 0);
    CHECK(after.sa_handler == before.sa_handler);
}

TEST_CASE("the watchdog turns a Ctrl-C into a stop request") {
    ash::cli::clear_interrupt();
    const ash::cli::InterruptHandler handler;
    std::stop_source stop;

    const ash::cli::TurnWatchdog watchdog{stop};
    CHECK_FALSE(stop.stop_requested());

    ::raise(SIGINT);

    // Through the watchdog and not straight from the handler, so this is also
    // the check that the polling thread is really running.
    CHECK(wait_for([&] { return stop.stop_requested(); }, std::chrono::milliseconds{500}));
    ash::cli::clear_interrupt();
}

TEST_CASE("the watchdog leaves an uninterrupted run alone") {
    ash::cli::clear_interrupt();
    std::stop_source stop;
    const ash::cli::TurnWatchdog watchdog{stop};

    // Several polls' worth, so a watchdog that stopped the run unconditionally
    // has had every chance to do it.
    std::this_thread::sleep_for(std::chrono::milliseconds{100});

    CHECK_FALSE(stop.stop_requested());
}

TEST_CASE("nothing is watching the flag once the turn it belonged to is over") {
    // This is what keeps a Ctrl-C aimed at one turn from ending the next one:
    // the watchdog exists for a turn and no longer, so a signal arriving between
    // turns has nothing to act on. The session clears the flag before opening a
    // turn, which is what stops the leftover from being picked up there.
    ash::cli::clear_interrupt();
    const ash::cli::InterruptHandler handler;
    std::stop_source stop;

    {
        const ash::cli::TurnWatchdog watchdog{stop};
        CHECK_FALSE(stop.stop_requested());
    }

    ::raise(SIGINT);
    std::this_thread::sleep_for(std::chrono::milliseconds{50});

    CHECK(ash::cli::g_interrupted != 0);
    CHECK_FALSE(stop.stop_requested());
    ash::cli::clear_interrupt();
}
